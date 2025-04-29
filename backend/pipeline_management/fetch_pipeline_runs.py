from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import psycopg2
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from db_util import get_db_connection


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.FileHandler("dashboard_backend.log"), logging.StreamHandler()]
)
logger = logging.getLogger("app")


app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify the actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PipelineRun(BaseModel):
    run_id: int
    start_time: str
    stage_1_status: str
    stage_1_start: Optional[str]
    stage_1_end: Optional[str]
    stage_2_status: str
    stage_2_start: Optional[str]
    stage_2_end: Optional[str]
    stage_3_status: str
    stage_3_start: Optional[str]
    stage_3_end: Optional[str]
    stage_4_status: str
    stage_4_start: Optional[str]
    stage_4_end: Optional[str]
    stage_5_status: str
    stage_5_start: Optional[str]
    stage_5_end: Optional[str]
    stage_6_status: str
    stage_6_start: Optional[str]
    stage_6_end: Optional[str]
    stage_7_status: str
    stage_7_start: Optional[str]
    stage_7_end: Optional[str]

# def get_db_connection():
#     return psycopg2.connect(
#         "dbname=postgres user=postgres password=postgres host=localhost"
#     )

@app.get("/")
def something():
    return "hello"

@app.get("/pipeline_runs", response_model=List[PipelineRun])
def get_pipeline_runs():
    logger.info("Fetching pipeline runs")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM pipeline_runs ORDER BY start_time DESC')
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        logger.error(f"Database error in /pipeline_runs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error")
    
    # Convert to list of dictionaries
    # column_names = ['run_id', 'start_time', 'stage_1_status', 'stage_2_status', 
    #                 'stage_3_status', 'stage_4_status', 'stage_5_status', 
    #                 'stage_6_status', 'stage_7_status']
    

    column_names = ['run_id', 'start_time', 'stage_1_status', 'stage_1_start', 'stage_1_end',
                    'stage_2_status', 'stage_2_start', 'stage_2_end',
                    'stage_3_status', 'stage_3_start', 'stage_3_end',
                    'stage_4_status', 'stage_4_start', 'stage_4_end',
                    'stage_5_status', 'stage_5_start', 'stage_5_end',
                    'stage_6_status', 'stage_6_start', 'stage_6_end',
                    'stage_7_status', 'stage_7_start', 'stage_7_end'
    ]
    result = []

    for row in rows:
        # datetime_str = str(row[1])
        new_row = [str(v) if str(type(v)) == "<class 'datetime.datetime'>" else v for v in row]
        # new_row = [row[0]] + [datetime_str] + list(row[2:])
        result.append(dict(zip(column_names, new_row)))

    logger.info(f"Returned {len(result)} pipeline runs")
    
    return result

# Add this block to run the server when the file is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
