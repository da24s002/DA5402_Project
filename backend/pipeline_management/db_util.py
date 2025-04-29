
import psycopg2
from db_config import user, password, host, dbname
# Database connection function
def get_db_connection():
    return psycopg2.connect(
        f"dbname={dbname} user={user} password={password} host={host}"
    )

# Create a new run in the database
def create_new_run():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
    INSERT INTO pipeline_runs (
        stage_1_status, stage_2_status, stage_3_status, stage_4_status,
        stage_5_status, stage_6_status, stage_7_status
    ) VALUES (
        'waiting', 'waiting', 'waiting', 'waiting',
        'waiting', 'waiting', 'waiting'
    ) RETURNING run_id;
    ''')
    run_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return run_id

def initialize_database():
    conn = get_db_connection()
    cur = conn.cursor()
    
    create_table_sql = '''
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        run_id SERIAL PRIMARY KEY,
        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        stage_1_status VARCHAR(20) DEFAULT 'waiting',
        stage_1_start TIMESTAMP,
        stage_1_end TIMESTAMP,
        stage_2_status VARCHAR(20) DEFAULT 'waiting',
        stage_2_start TIMESTAMP,
        stage_2_end TIMESTAMP,
        stage_3_status VARCHAR(20) DEFAULT 'waiting',
        stage_3_start TIMESTAMP,
        stage_3_end TIMESTAMP,
        stage_4_status VARCHAR(20) DEFAULT 'waiting',
        stage_4_start TIMESTAMP,
        stage_4_end TIMESTAMP,
        stage_5_status VARCHAR(20) DEFAULT 'waiting',
        stage_5_start TIMESTAMP,
        stage_5_end TIMESTAMP,
        stage_6_status VARCHAR(20) DEFAULT 'waiting',
        stage_6_start TIMESTAMP,
        stage_6_end TIMESTAMP,
        stage_7_status VARCHAR(20) DEFAULT 'waiting',
        stage_7_start TIMESTAMP,
        stage_7_end TIMESTAMP
    );
    '''
    cur.execute(create_table_sql)
    conn.commit()
    print("Created pipeline_runs table")
    conn.close()

# Update stage status
def update_stage_status(run_id, stage_number, status, start_time=None, end_time=None):
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Update status and timestamps
    if status == 'running':
        cur.execute(f'''
        UPDATE pipeline_runs
        SET stage_{stage_number}_status = %s,
            stage_{stage_number}_start = CURRENT_TIMESTAMP
        WHERE run_id = %s;
        ''', (status, run_id))
    elif status in ('complete', 'failed'):
        cur.execute(f'''
        UPDATE pipeline_runs
        SET stage_{stage_number}_status = %s,
            stage_{stage_number}_end = CURRENT_TIMESTAMP
        WHERE run_id = %s;
        ''', (status, run_id))
    
    # If failed, update all subsequent stages to failed
    if status == 'failed':
        for next_stage in range(stage_number + 1, 8):
            cur.execute(f'''
            UPDATE pipeline_runs
            SET stage_{next_stage}_status = 'failed'
            WHERE run_id = %s;
            ''', (run_id,))
    
    conn.commit()
    conn.close()


def set_stages_not_triggered(run_id, start_stage, end_stage):
    conn = get_db_connection()
    cur = conn.cursor()
    for stage in range(start_stage, end_stage + 1):
        cur.execute(f'''
            UPDATE pipeline_runs
            SET stage_{stage}_status = %s
            WHERE run_id = %s;
        ''', ('not_triggered', run_id))
    conn.commit()
    conn.close()