import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set a smaller context for all plots to reduce text size
sns.set_theme(style="whitegrid", context="notebook", palette="pastel")

# Function to add a frame to the plot
def add_plot_frame(ax, color='black', linewidth=2, padding=0.01):
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(linewidth)

    # Adjust the subplots to add padding for the frame
    plt.subplots_adjust(left=padding, right=1-padding, top=1-padding, bottom=padding)


# Define the function to plot Job Profile Match with a styled frame and smaller size
def plot_job_profile_match(job_profile_match_score, plot_size):
    fig, ax = plt.subplots(figsize=plot_size)
    profile_match_labels = ['Match', 'No Match']
    profile_match_sizes = [job_profile_match_score, 100 - job_profile_match_score]
    profile_match_colors = ['lightgreen', 'lightcoral']
    profile_match_explode = (0.1, 0)  # explode the first slice for emphasis
    
    # Customize text size with `autopct` and `textprops`
    ax.pie(profile_match_sizes, labels=profile_match_labels, colors=profile_match_colors, explode=profile_match_explode,
           autopct=lambda p: '{:.0f}%'.format(p) if p > 0 else '',  # Only show if percentage > 0
           shadow=True, startangle=140, textprops={'fontsize': 4})
    
    return fig

# Define the function to plot Educational Qualifications with a styled frame and smaller size
def plot_educational_qualifications(education_match, plot_size):
    fig, ax = plt.subplots(figsize=plot_size)
    education_labels = ['Required Qualification']
    education_values = [1 if education_match else 0]
    education_colors = ['lightblue' if education_match else 'lightcoral']

    sns.barplot(x=education_values, y=education_labels, palette=education_colors, ax=ax, orient='h')
    ax.set_xlabel('Match', fontsize=9)  # Adjust font size here
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'], fontsize=8)  # Adjust font size here

    ax.set_yticklabels(education_labels, fontsize=8)  # Adjust font size here

    return fig


# Define the function to plot Skills and Proficiencies with a styled frame and smaller size
def plot_skills_and_proficiencies(skills_scores, plot_size):
    labels = np.array(list(skills_scores.keys()))
    stats = np.array(list(skills_scores.values()))

    fig, ax = plt.subplots(figsize=plot_size)
    ax.barh(labels, stats, color='skyblue')
    ax.set_xlabel('Match Level', fontsize=4)  # Adjust font size here
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_xticklabels(np.arange(0, 110, 20), fontsize=4)  # Adjust font size here

    ax.set_yticklabels(labels, fontsize=4)  # Adjust font size here

    return fig

def plot_spyder(plot_size):
    skills_scores = {
        "Logistics": 0.8,
        "International Trade": 0.9,
        "Legal Advisory": 0.7,
        "Fiscal Advisory": 0.5,
        "Analytical Skills": 0.3,
    }

    labels = np.array(list(skills_scores.keys()))
    stats = np.array(list(skills_scores.values()))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats = np.concatenate((stats,[stats[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=plot_size, subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color='green', alpha=0.25)
    ax.plot(angles, stats, color='green', linewidth=2)  
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=4)
    ax.set_yticklabels([], fontsize=4)



# Define the function to plot Certifications and Regulatory Knowledge with a styled frame and smaller size
def plot_certifications_and_regulatory_knowledge(certifications_presence, certifications_labels, plot_size):
    fig, ax = plt.subplots(figsize=plot_size)
    certifications_colors = ['lightgreen' if val else 'lightcoral' for val in certifications_presence]

    sns.barplot(x=certifications_presence, y=certifications_labels, palette=certifications_colors, ax=ax, orient='h')
    ax.set_xlabel('Presence', fontsize=9)  # Adjust font size here
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'], fontsize=8)  # Adjust font size here

    # Set the y-tick labels font size
    ax.set_yticklabels(certifications_labels, fontsize=10)  # Adjust font size here

    return fig

# scripts.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def display_similarity_bar(json_params):
    score = json_params['score']
    score_percentage = f"{score:.2%}"
    st.progress(score)
    st.write(f"Similarity Score: {score_percentage}")

def display_education_comparison(json_params):
    cv_education = json_params['cv_education']
    job_offer_education = json_params['job_offer_education']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('### Education for CV')
        for education in cv_education:
            st.markdown(f"- {education}")

    with col2:
        st.markdown('### Required Education by Job Offer')
        for education in job_offer_education:
            st.markdown(f"- {education}")

def display_language_comparison(json_params):
    languages_cv = json_params['languages_cv']
    languages_job_requirement = json_params['languages_job_requirement']

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('#### CV Languages')
        for language, flag in languages_cv.items():
            st.write(f"{flag} {language}")

    with col2:
        st.markdown('#### Job Requirement Languages')
        for language, flag in languages_job_requirement.items():
            st.write(f"{flag} {language}")

def create_matplotlib_chart(json_params):
    data = pd.DataFrame(json_params['data'])
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=data, x='x', y='y')
    plt.tight_layout()
    st.pyplot(plt.gcf())

def create_seaborn_chart(json_params):
    data = pd.DataFrame(json_params['data'])
    plt.figure(figsize=(6, 4))
    sns.barplot(data=data, x='x', y='y')
    plt.tight_layout()
    st.pyplot(plt.gcf())

# Mock value for job profile match score (suppose the candidate's profile matches 70% with the job profile)
job_profile_match_score = 70

# Mock boolean for education match (suppose the candidate has the required educational qualifications)
education_match = True

# Mock dictionary for skills scores (each skill has a match score out of 1)
skills_scores = {
    "Logistics": 0.8,  # 80% match
    "International Trade": 0.9,  # 90% match
    "Legal Advisory": 0.7,  # 70% match
    "Fiscal Advisory": 0.5,  # 50% match
    "IT": 0.3,  # 30% match
}

# Mock list for certifications presence (True if the candidate has the certification, False otherwise)
certifications_presence = [True, False, True, False, True]

# Mock labels for certifications (corresponding to the above presence/absence list)
certifications_labels = [
    "Certified Supply Chain Professional",
    "Certified in Logistics, Transportation and Distribution",
    "Project Management Professional",
    "Has Logistics Bachelor",
    "Occupational Safety and Health Administration Certification"
]

def toggle_charts(json_params):
    show_charts = json_params['show_charts']
    plot_size = (1,1)
    if show_charts:
        #  with st.container:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Job Profile Match")
            st.write("This pie chart shows the percentage match of the candidate's job profile with the job description.")
            # Call the function and use st.pyplot to render the figure
            st.pyplot(plot_job_profile_match(job_profile_match_score, plot_size=plot_size))

        with col2:
            st.subheader("Educational Qualifications")
            st.write("The bar chart indicates whether the candidate has the educational qualifications required for the job.")
            # Call the function and use st.pyplot to render the figure
            st.pyplot(plot_educational_qualifications(education_match, plot_size=plot_size))

        with col1:
            st.subheader("Skills and Proficiencies")
            st.write("The radar chart visualizes the candidate's skill levels across various domains required for the job.")
            # Call the function and use st.pyplot to render the figure
            st.pyplot(plot_skills_and_proficiencies(skills_scores, plot_size=plot_size))

        with col2:
            st.subheader("Certifications and Regulatory Knowledge")
            st.write("This bar chart shows whether the candidate possesses the specific certifications or knowledge areas important for the position.")
            # Call the function and use st.pyplot to render the figure
            st.pyplot(plot_certifications_and_regulatory_knowledge(certifications_presence, certifications_labels, plot_size=plot_size))


