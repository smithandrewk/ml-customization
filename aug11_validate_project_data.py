from utils import get_db_connection
from dotenv import load_dotenv
import json

load_dotenv()

conn = get_db_connection()

cursor = conn.cursor()

cursor.execute("SELECT project_id,project_name,participant_id FROM projects WHERE project_name = 'unk1'")

project_id, project_name, participant_id = cursor.fetchone()

print(f"Project ID: {project_id}, Project Name: {project_name}, Participant ID: {participant_id}")

# Get number of sessions for the project
cursor.execute(f"SELECT COUNT(*) FROM sessions WHERE project_id = {project_id}")
num_sessions = cursor.fetchone()[0]

# Get length and number of bouts of each session
cursor.execute(f"SELECT session_id, start_ns, stop_ns, bouts FROM sessions WHERE project_id = {project_id} AND verified=1 AND (keep IS NULL OR keep != 0)")
sessions = cursor.fetchall()


total_duration_seconds = 0
total_bouts = 0
total_bout_duration = 0

for session_id, start_ns, stop_ns, bouts in sessions:
    bouts = json.loads(bouts)
    total_bouts += len(bouts)
    total_bout_duration += sum((bout['end'] - bout['start']) * 1e-9 for bout in bouts)
    duration_ns = stop_ns - start_ns
    duration_seconds = duration_ns / 1e9 / 3600
    total_duration_seconds += duration_seconds
    print(f"Session ID: {session_id}, Duration: {duration_seconds:.2f} hours, Number of bouts: {len(bouts)}, Average bout duration: {sum((bout['end'] - bout['start']) * 1e-9 for bout in bouts) / len(bouts) if len(bouts) > 0 else 0:.2f} seconds")

# Print if there are any unverified sessions
cursor.execute(f"SELECT COUNT(*) FROM sessions WHERE project_id = {project_id} AND verified= 0 AND (keep IS NULL OR keep != 0)")
num_unverified = cursor.fetchone()[0]
if num_unverified > 0:
    print(f"Number of unverified sessions for project {project_name}: {num_unverified}")

# print total duration in hours , total_bouts, and total bout duration
print(f"Number of sessions for project {project_name}: {len(sessions)}")
print(f"Total duration for project {project_name}: {total_duration_seconds:.2f} hours")
print(f"Total number of bouts for project {project_name}: {total_bouts}")
print(f"Total bout duration for project {project_name}: {total_bout_duration:.2f} seconds")