import psycopg2
import os
import streamlit as st
import datetime
from psycopg2.extensions import adapt, register_adapter, AsIs
import numpy as np

# --- Database Connection ---

def get_db_connection():
    """Establishes a connection to the PostgreSQL database using Streamlit secrets."""
    try:
        conn = psycopg2.connect(
            dbname=st.secrets.get("POSTGRES_DB", "postgres"), # Default DB name
            user=st.secrets.get("POSTGRES_USER", "postgres"), # Default user
            password=st.secrets.get("POSTGRES_PASSWORD", "password"), # !! CHANGE THIS DEFAULT IN secrets.toml !!
            host=st.secrets.get("POSTGRES_HOST", "localhost"), # Default to localhost
            port=st.secrets.get("POSTGRES_PORT", "5432") # Default port
        )
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        # Check specifically for missing secrets if it's an AttributeError
        if isinstance(error, AttributeError) and 'st.secrets' in str(error):
             st.error("Database secrets not found. Make sure you have a .streamlit/secrets.toml file configured.")
        else:
             st.error(f"Database Connection Error: {error}")
        return None
    except KeyError as e:
        st.error(f"Missing database secret: {e}. Please check your .streamlit/secrets.toml file.")
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
        except (Exception, psycopg2.DatabaseError) as error:
            st.error(f"Failed to create table: {error}")
        finally:
            conn.close()

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
        except (Exception, psycopg2.DatabaseError) as error:
            st.error(f"Failed to log prediction to database: {error}")
        finally:
            conn.close()

# --- Optional: Adapter for NumPy arrays if needed later (not strictly required for tobytes) ---
# def adapt_numpy_array(numpy_array):
#     return AsIs(adapt(numpy_array.tolist())) # Example adapter
# register_adapter(np.ndarray, adapt_numpy_array) 