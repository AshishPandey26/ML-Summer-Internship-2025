import streamlit as st
import os
from datetime import datetime
from utils import load_feedback, save_feedback, analyze_sentiment, get_employee_feedback, FEEDBACK_FILE

# Ensure the 'data' directory exists. This is crucial for file operations.
if not os.path.exists('data'):
    os.makedirs('data')

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Employee Feedback System",
    layout="centered", # 'centered' or 'wide'
    initial_sidebar_state="auto" # 'auto', 'expanded', or 'collapsed'
)

# --- Session State Initialization ---
# Initialize a session state variable to keep track of how many times feedback
# has been submitted for each employee. This persists across Streamlit reruns.
if 'feedback_counters' not in st.session_state:
    st.session_state['feedback_counters'] = {}

# Initialize input fields in session state to allow clearing them after submission
if 'employee_name_input' not in st.session_state:
    st.session_state.employee_name_input = ""
if 'feedback_text_input' not in st.session_state:
    st.session_state.feedback_text_input = ""

# --- Title and Introduction ---
st.title("Employee Feedback System")
st.markdown("""
Welcome to the Employee Feedback System!
Enter feedback for employees, and after the 4th submission for a person,
you'll see past feedback entries for reference.
""")
st.markdown("---")

# --- Feedback Input Section ---
st.header("Submit New Feedback")

# Text input for employee name.
# We now use the 'value' parameter to link the input to the session state variable.
employee_name = st.text_input("Employee Name", key="employee_name_input", value=st.session_state.employee_name_input)
# Text area for feedback.
# We now use the 'value' parameter to link the input to the session state variable.
feedback_text = st.text_area("Feedback", key="feedback_text_input", value=st.session_state.feedback_text_input)

# Load all existing feedback from the JSON file
all_feedback = load_feedback()

# Get feedback specific to the currently entered employee name
current_employee_past_feedback = get_employee_feedback(employee_name, all_feedback)
# Count how many past feedbacks exist for this employee
num_past_feedbacks = len(current_employee_past_feedback)

# Update the feedback counter for the current employee in session state.
# This counter reflects the number of *already submitted* feedbacks.
if employee_name:
    st.session_state['feedback_counters'][employee_name.lower()] = num_past_feedbacks

# --- Suggestion Logic (now "Previous Feedback" Logic) ---
# Display previous feedback if the employee name is provided and they have 4 or more
# existing feedbacks (meaning the current submission will be the 5th or more).
if employee_name and st.session_state['feedback_counters'].get(employee_name.lower(), 0) >= 4:
    # Changed heading to reflect that these are previous feedback entries
    st.subheader(f"📝 Previous Feedback for {employee_name}:")
    if current_employee_past_feedback:
        # Display the last 3 feedbacks as previous entries. You can adjust this number.
        # We reverse to show most recent first among the previous entries.
        for i, fb in enumerate(reversed(current_employee_past_feedback[-3:])):
            st.info(f"**Feedback {num_past_feedbacks - i} ({fb['sentiment']}):** {fb['feedback_text']}")
    else:
        st.info("No past feedback available for this employee.")

# --- Submit Button Logic ---
if st.button("Submit Feedback", key="submit_feedback_button", type="primary"):
    if employee_name.strip() and feedback_text.strip(): # Check if inputs are not just whitespace
        # Analyze the sentiment of the new feedback
        sentiment = analyze_sentiment(feedback_text)
        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create a new feedback entry dictionary
        new_feedback_entry = {
            "employee_name": employee_name.strip(), # Clean up whitespace
            "feedback_text": feedback_text.strip(), # Clean up whitespace
            "timestamp": timestamp,
            "sentiment": sentiment
        }

        # Add the new entry to the list of all feedback
        all_feedback.append(new_feedback_entry)
        # Save the updated list back to the JSON file
        save_feedback(all_feedback)

        st.success(f"Feedback submitted for **{employee_name.strip()}**! Sentiment: **{sentiment}**")

        # Clear the input fields by updating their session state values
        st.session_state.employee_name_input = ""
        st.session_state.feedback_text_input = ""
        # Force a rerun of the Streamlit app to update the display,
        # especially the "All Past Feedback" section and counters.
        # This rerun will now pick up the cleared values for the input widgets.
        st.experimental_rerun()
    else:
        st.warning("Please enter both employee name and feedback before submitting.")

st.markdown("---")

# --- Display All Past Feedback Section ---
st.header("All Past Feedback Records")
if all_feedback:
    # Display feedback in reverse chronological order (most recent first)
    for fb in reversed(all_feedback):
        st.markdown(f"**Employee:** {fb['employee_name']}")
        st.markdown(f"**Feedback:** {fb['feedback_text']}")
        st.markdown(f"**Sentiment:** <span style='color:{'green' if fb['sentiment'] == 'POSITIVE' else 'red'};'>**{fb['sentiment']}**</span>", unsafe_allow_html=True)
        st.markdown(f"**Timestamp:** {fb['timestamp']}")
        st.markdown("---")
else:
    st.info("No feedback has been submitted yet. Start by adding some!")
