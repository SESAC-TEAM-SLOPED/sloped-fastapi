import os

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json
from psycopg2 import Error
import psycopg2
from fastapi import FastAPI, HTTPException


def get_facility_by_id(facility_id, review_id):
    conn = None

    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST
        )

        with conn.cursor() as cur:
            query = """
            SELECT
                f.id,
                f.name,
                f.address,
                f.accessibility_description,
                fr.content as recent_review,
                ST_X(f.point::geometry) as longitude,
                ST_Y(f.point::geometry) as latitude
            FROM
                facility f
            LEFT JOIN
                facility_review fr ON f.id = fr.facility_id AND fr.id = %s
            WHERE
                f.id = %s;
            """

            cur.execute(query, (review_id, facility_id))
            facility = cur.fetchone()
            print(f"Retrieved facility: {facility}")
            return facility
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
        return None
    finally:
        if conn:
            conn.close()


def get_facility_info(facility_id, review_id):
    facility = get_facility_by_id(facility_id, review_id)

    print(f"Facility data: {facility}")

    if not facility:
        print("Facility not found")
        return None

    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")

    if facility[3] is None:  # If accessibility_description is None
        messages = PROMPT_NO_DESCRIPTION.format_messages(
            facility_id=facility[0],
            facility_name=facility[1],
            facility_address=facility[2],
            reviews=facility[4] or "리뷰 없음",
            facility_longitude=facility[5],
            facility_latitude=facility[6]
        )
    else:  # If accessibility_description exists
        messages = PROMPT_WITH_DESCRIPTION.format_messages(
            facility_id=facility[0],
            facility_name=facility[1],
            facility_address=facility[2],
            existing_description=facility[3],
            recent_review=facility[4] or "최근 리뷰 없음",
            facility_longitude=facility[5],
            facility_latitude=facility[6]
        )

    result = llm.invoke(messages)
    print(f"LLM response: {result.content}")

    try:
        json_str = result.content.strip()
        if json_str.startswith('```json'):
            json_str = json_str[7:-3].strip()

        parsed_result = json.loads(json_str)

        parsed_result['id'] = int(parsed_result['id'])
        parsed_result['점수'] = int(parsed_result['점수'])

        print(f"Parsed and verified result: {parsed_result}")
        return parsed_result
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error processing LLM response: {e}")
        print(f"Problematic JSON string: {json_str}")
        return {
            "id": facility[0],
            "점수": 0,
            "설명": "정보가 불충분하여 평가할 수 없습니다."
        }


def save_facility_review(facility_id, review_id, accessibility_score, accessibility_description):
    conn = None

    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST
        )

        with conn.cursor() as cur:
            # Update existing review
            update_review_query = """
            UPDATE facility_review 
            SET accessibility_score = %s, updated_at = NOW()
            WHERE id = %s AND facility_id = %s
            """
            cur.execute(update_review_query, (accessibility_score, review_id, facility_id))

            # Update facility with average score and new description
            update_facility_query = """
            UPDATE facility
            SET average_accessibility_score = (
                SELECT AVG(accessibility_score)::bigint
                FROM facility_review
                WHERE facility_id = %s
            ),
            accessibility_description = %s
            WHERE id = %s
            """
            cur.execute(update_facility_query, (facility_id, accessibility_description, facility_id))

            conn.commit()

            print(f"Updated facility review with id: {review_id}")
            print(f"Updated facility accessibility_description: {accessibility_description}")
            return True
    except (Exception, Error) as error:
        print("Error while updating facility review", error)
        return False
    finally:
        if conn:
            conn.close()


TEMPLATE_NO_DESCRIPTION = """
    시설 정보:
    - 시설ID: {facility_id}
    - 이름: {facility_name}
    - 주소: {facility_address}
    - 위치: 위도 {facility_latitude}, 경도 {facility_longitude}

    리뷰: {reviews}

    Instruction: 위 시설 정보와 리뷰를 고려하여 답변하세요.

    Question: 위 시설이 장애인에게 얼마나 친화적인지 평가하고, 0에서 100 사이의 점수를 매겨주세요. 
    또한 시설의 접근성에 대해 100자 이내로 설명해주세요.

    Answer: 다음 JSON 형식으로 답변해주세요:
    {{
        "id": {facility_id},
        "점수": 0에서 100 사이의 평가 점수,
        "설명": "100자 이내의 접근성 설명 (가능하면 '~입니다' 체를 사용해주세요)"
    }}

    주의: 
    1. 반드시 위에서 제공된 실제 시설 정보와 id를 사용하세요. 
    2. 임의의 정보를 생성하지 마세요.
    3. 리뷰 내용을 모두 고려하여 평가하세요.
    4. '설명' 부분은 가능하면 '~입니다' 체를 사용해주세요. 하지만 자연스러운 표현이 우선입니다. '~임' 체는 사용하지 마세요. 
    """

TEMPLATE_WITH_DESCRIPTION = """
    시설 정보:
    - 시설ID: {facility_id}
    - 이름: {facility_name}
    - 주소: {facility_address}
    - 위치: 위도 {facility_latitude}, 경도 {facility_longitude}

    기존 접근성 설명: {existing_description}

    최근 리뷰: {recent_review}

    Instruction: 위 시설 정보, 기존 접근성 설명, 그리고 최근 리뷰를 고려하여 답변하세요.

    Question: 위 시설이 장애인에게 얼마나 친화적인지 평가하고, 0에서 100 사이의 점수를 매겨주세요. 
    또한 기존 접근성 설명을 최근 리뷰 내용을 반영하여 100자 이내로 업데이트해주세요.

    Answer: 다음 JSON 형식으로 답변해주세요:
    {{
        "id": {facility_id},
        "점수": 0에서 100 사이의 평가 점수,
        "설명": "100자 이내의 업데이트된 접근성 설명 (가능하면 '~입니다' 체를 사용해주세요)"
    }}

    주의: 
    1. 반드시 위에서 제공된 실제 시설 정보와 id를 사용하세요. 
    2. 임의의 정보를 생성하지 마세요.
    3. 기존 접근성 설명과 최근 리뷰 내용을 모두 고려하여 평가하세요.
    4. '설명' 부분은 가능하면 '~입니다' 체를 사용해주세요. 하지만 자연스러운 표현이 우선입니다. '~임' 체는 사용하지 마세요. 
    """

PROMPT_NO_DESCRIPTION = ChatPromptTemplate.from_template(TEMPLATE_NO_DESCRIPTION)
PROMPT_WITH_DESCRIPTION = ChatPromptTemplate.from_template(TEMPLATE_WITH_DESCRIPTION)


load_dotenv()
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY")


app = FastAPI()


@app.get("/facility-reviews/assessment")
async def process_facility(facility_id: int, review_id: int):
    facility_info = get_facility_info(facility_id, review_id)

    if facility_info:
        success = save_facility_review(
            facility_id,
            review_id,
            facility_info['점수'],
            facility_info['설명']
        )
        if success:
            return {"status": "success", "message": "RAG 검색 성공"}
        else:
            raise HTTPException(status_code=500, detail={"status": "fail", "message": "시설과 시설리뷰 업데이트에 실패하였습니다"})
    else:
        raise HTTPException(status_code=404, detail={"status": "fail", "message": "존재하지 않는 시설입니다"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)