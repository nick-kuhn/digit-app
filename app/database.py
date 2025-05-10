import psycopg2
import os
import streamlit as st
import datetime
from psycopg2.extensions import adapt, register_adapter, AsIs
import numpy as np

# --- Database Connection ---

def get_db_connection():
    """Establishes a connection to the PostgreSQL database using environment variables."""
    try:
        db_name = os.getenv("POSTGRES_DB", "postgres")
        db_user = os.getenv("POSTGRES_USER", "postgres")
        db_password = os.getenv("POSTGRES_PASSWORD") # No default for password for security
        db_host = os.getenv("POSTGRES_HOST", "localhost") # Default for non-Compose local dev
        db_port = os.getenv("POSTGRES_PORT", "5432")

        if not db_password:
            st.error("Database password (POSTGRES_PASSWORD) is not set as an environment variable.")
            return None

        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host, # For Docker Compose, this will be 'db'
            port=db_port
        )
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        # Log the error to Streamlit, which will be visible in container logs if it happens early
        st.error(f"Database Connection Error: {error}. Check DB service and env vars.")
        # Optionally, print to stderr as well for direct container log visibility
        print(f"Database Connection Error: {error}") 
        return None

# --- Table Creation ---

def create_table():
    """Creates the prediction_logs table if it doesn't exist."""
    sql = """
    CREATE TABLE IF NOT EXISTS prediction_logs (
        log_id SERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        predicted_digit INTEGER,
        confidence FLOAT,
        true_label INTEGER,
        image_data BYTEA
    );
    """
    conn = get_db_connection()
    if conn is not None:
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                conn.commit()
            # st.info("Database table 'prediction_logs' checked/created.") # Optional: success message
        except (Exception, psycopg2.DatabaseError) as error:
            st.error(f"Failed to create table 'prediction_logs': {error}")
            print(f"Failed to create table 'prediction_logs': {error}")
        finally:
            conn.close()
    else:
        st.error("Could not create table: No database connection.")
        print("Could not create table: No database connection.")

# --- Logging Function ---

def log_prediction_db(predicted_digit, confidence, true_label, image_data_rgba):
    """Logs prediction details and raw image data to the database."""
    sql = """
    INSERT INTO prediction_logs (timestamp, predicted_digit, confidence, true_label, image_data)
    VALUES (%s, %s, %s, %s, %s);
    """
    conn = get_db_connection()
    if conn is not None:
        try:
            image_bytes = image_data_rgba.tobytes() if image_data_rgba is not None else None
            timestamp = datetime.datetime.now()

            with conn.cursor() as cur:
                cur.execute(sql, (timestamp, predicted_digit, float(confidence), true_label, image_bytes))
                conn.commit()
            # st.info("Prediction logged to database.") # Optional: success message
        except (Exception, psycopg2.DatabaseError) as error:
            st.error(f"Failed to log prediction to database: {error}")
            print(f"Failed to log prediction to database: {error}")
        finally:
            conn.close()
    else:
        st.error("Could not log prediction: No database connection.")
        print("Could not log prediction: No database connection.")

# --- Optional: Adapter for NumPy arrays if needed later (not strictly required for tobytes) ---
# def adapt_numpy_array(numpy_array):
#     return AsIs(adapt(numpy_array.tolist())) # Example adapter
# register_adapter(np.ndarray, adapt_numpy_array) 