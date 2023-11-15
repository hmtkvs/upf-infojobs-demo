import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set a smaller context for all plots to reduce text size
sns.set_theme(style="whitegrid", context="notebook", palette="pastel")

# Define the function to plot Job Profile Match with a styled frame and smaller size
def plot_job_profile_match(job_profile_match_score, plot_size=(4, 2)):
    fig, ax = plt.subplots(figsize=plot_size)
    profile_match_labels = 'Match', 'No Match'
    profile_match_sizes = [job_profile_match_score, 100 - job_profile_match_score]
    profile_match_colors = ['lightgreen', 'lightcoral']
    profile_match_explode = (0.1, 0)
    
    ax.pie(profile_match_sizes, explode=profile_match_explode, labels=profile_match_labels, colors=profile_match_colors,
           autopct='%1.1f%%', shadow=True, startangle=140)
    ax.set_title('Job Profile Match')
    
    return fig

# Define the function to plot Educational Qualifications with a styled frame and smaller size
def plot_educational_qualifications(education_match, plot_size=(4, 2)):
    fig, ax = plt.subplots(figsize=plot_size)
    education_labels = ['Required Qualification']
    education_values = [1 if education_match else 0]
    education_colors = ['lightblue' if education_match else 'lightcoral']
    
    sns.barplot(x=education_values, y=education_labels, palette=education_colors, ax=ax, orient='h')
    ax.set_title('Educational Qualifications')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Match')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'])
    
    return fig

# Define the function to plot Skills and Proficiencies with a styled frame and smaller size
def plot_skills_and_proficiencies(skills_scores, plot_size=(4, 4)):
    labels = np.array(list(skills_scores.keys()))
    stats = np.array(list(skills_scores.values()))
    
    fig, ax = plt.subplots(figsize=plot_size)
    ax.barh(labels, stats, color='skyblue')
    ax.set_title('Skills And Proficiencies')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Match Level')
    
    return fig

# Define the function to plot Certifications and Regulatory Knowledge with a styled frame and smaller size
def plot_certifications_and_regulatory_knowledge(certifications_presence, certifications_labels, plot_size=(4, 2)):
    fig, ax = plt.subplots(figsize=plot_size)
    certifications_colors = ['lightgreen' if val else 'lightcoral' for val in certifications_presence]
    
    sns.barplot(x=certifications_presence, y=certifications_labels, palette=certifications_colors, ax=ax, orient='h')
    ax.set_title('Certifications And Knowledge')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Presence')
    
    return fig
