# app/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
from datetime import datetime, timedelta
import traceback
import base64
import io

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules with error handling
try:
    from extract.comment_extractor import CommentExtractor
    comment_extractor_available = True
except ImportError as e:
    st.error(f"Error importing CommentExtractor: {e}")
    comment_extractor_available = False
    # Define fallback class
    class CommentExtractor:
        def extract_from_directory(self, *args, **kwargs):
            return []

try:
    from extract.commit_extractor import CommitExtractor
    commit_extractor_available = True
except ImportError as e:
    st.error(f"Error importing CommitExtractor: {e}")
    commit_extractor_available = False
    # Define fallback class
    class CommitExtractor:
        def __init__(self, *args, **kwargs):
            pass
        def extract_commits(self, *args, **kwargs):
            return []

try:
    from model.emotion_classifier import EmotionClassifier
    emotion_classifier_available = True
except ImportError as e:
    st.error(f"Error importing EmotionClassifier: {e}")
    emotion_classifier_available = False
    # Define fallback class
    class EmotionClassifier:
        def classify(self, text):
            # Return random emotion for demo purposes
            import random
            emotions = ['joy', 'pride', 'neutral', 'frustration', 'anger', 'confusion', 'urgency', 'concern']
            selected = random.choice(emotions)
            return {emotion: 0.1 for emotion in emotions}
        def process_comments(self, comments):
            return []

# Helper functions for additional features
def generate_download_link(df, filename, text):
    """Generate a download link for a DataFrame as CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return f'<a href="{href}" download="{filename}" target="_blank">{text}</a>'

def generate_json_download_link(data, filename, text):
    """Generate a download link for data as JSON."""
    json_str = json.dumps(data, indent=2, default=str)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'data:file/json;base64,{b64}'
    return f'<a href="{href}" download="{filename}" target="_blank">{text}</a>'

def filter_data_by_text(data, search_text, fields):
    """Filter data by search text across specified fields."""
    if not search_text:
        return data
    
    search_text = search_text.lower()
    filtered = []
    
    for item in data:
        match = False
        for field in fields:
            if field in item and isinstance(item[field], str) and search_text in item[field].lower():
                match = True
                break
        if match:
            filtered.append(item)
    
    return filtered

def sort_data(data, sort_by, sort_order):
    """Sort data by specified field and order."""
    if not sort_by or sort_by not in data[0]:
        return data
    
    reverse = sort_order == "Descending"
    
    # Handle dates specially
    if sort_by == 'date':
        return sorted(data, key=lambda x: x.get(sort_by, ""), reverse=reverse)
    
    # Handle normal fields
    return sorted(data, key=lambda x: x.get(sort_by, ""), reverse=reverse)

# Initialize app
st.set_page_config(
    page_title="BERT CodeAuditor - Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("🧠 BERTCodeAuditor")
st.sidebar.markdown("## Code Analysis")

# Load user settings from session state
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'repo_path': '.',
        'max_commits': 100,
        'branch_name': 'main',
        'results_per_page': 10
    }

# Settings in sidebar
with st.sidebar.expander("Settings", expanded=False):
    st.session_state.settings['repo_path'] = st.text_input(
        "Default Repository Path", 
        value=st.session_state.settings['repo_path']
    )
    st.session_state.settings['max_commits'] = st.slider(
        "Default Max Commits", 
        10, 1000, 
        value=st.session_state.settings['max_commits']
    )
    st.session_state.settings['branch_name'] = st.text_input(
        "Default Branch Name", 
        value=st.session_state.settings['branch_name']
    )
    st.session_state.settings['results_per_page'] = st.slider(
        "Default Results Per Page", 
        5, 50, 
        value=st.session_state.settings['results_per_page']
    )

# Analysis type selector
analysis_type = st.sidebar.radio(
    "Analysis Type",
    ["Code Comments", "Commit Messages", "Combined Analysis", "Developer Profiles"]
)

# Main content
st.title("🧠 BERTCodeAuditor")
st.markdown("### Code Analysis from Code Comments & Commits")

# Initialize emotion classifier
@st.cache_resource
def load_emotion_classifier():
    if emotion_classifier_available:
        try:
            return EmotionClassifier()
        except Exception as e:
            st.error(f"Error initializing EmotionClassifier: {e}")
            return None
    return None

emotion_classifier = load_emotion_classifier()

# Code Comments Analysis
if analysis_type == "Code Comments":
    st.header("Code Comment Emotion Analysis")
    
    # Repository input
    repo_path = st.text_input("Repository Path", value=st.session_state.settings['repo_path'])
    
    # Advanced options
    with st.expander("Advanced Options"):
        ignore_dirs = st.text_input(
            "Directories to Ignore (comma-separated)", 
            value="node_modules,.git,venv,__pycache__,build,dist"
        ).split(',')
        
        file_types = st.multiselect(
            "File Types to Include",
            ['.py', '.js', '.ts', '.java', '.cs', '.cpp', '.c', '.rb', '.go', '.rs', '.html', '.css', '.php', '.swift', '.kt'],
            default=['.py', '.js', '.ts', '.java', '.cs', '.cpp', '.c']
        )
    
    # Scan button
    if st.button("Scan Repository for Comments"):
        with st.spinner("Scanning code comments..."):
            try:
                if not comment_extractor_available:
                    st.error("CommentExtractor module is not available. Please check your installation.")
                else:
                    # Extract comments
                    extractor = CommentExtractor()
                    comments = extractor.extract_from_directory(repo_path, ignore_dirs=ignore_dirs)
                    
                    # Filter by file type if specified
                    if file_types:
                        comments = [c for c in comments if any(c.get('file', '').endswith(ext) for ext in file_types)]
                    
                    if not comments:
                        st.warning(f"No comments found in repository: {repo_path}")
                    else:
                        st.success(f"Found {len(comments)} comments!")
                        
                        # Process with emotion classifier
                        if emotion_classifier:
                            processed_comments = []
                            for comment in comments:
                                # Make sure 'text' key exists
                                if 'text' in comment:
                                    try:
                                        emotion_scores = emotion_classifier.classify(comment['text'])
                                        top_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
                                        processed_comments.append({
                                            **comment,
                                            'emotion_scores': emotion_scores,
                                            'top_emotion': top_emotion
                                        })
                                    except Exception as e:
                                        # Include the comment anyway
                                        processed_comments.append(comment)
                            
                            # Store in session state for persistence
                            st.session_state.processed_comments = processed_comments
                            
                            # Display results
                            emotion_counts = {}
                            for comment in processed_comments:
                                if 'top_emotion' in comment:
                                    emotion = comment['top_emotion']
                                    if emotion not in emotion_counts:
                                        emotion_counts[emotion] = 0
                                    emotion_counts[emotion] += 1
                            
                            if emotion_counts:
                                # Create DataFrame for visualization
                                emotions_df = pd.DataFrame({
                                    'Emotion': list(emotion_counts.keys()),
                                    'Count': list(emotion_counts.values())
                                })
                                
                                # Create charts
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Plot emotions as bar chart
                                    fig = px.bar(
                                        emotions_df,
                                        x='Emotion',
                                        y='Count',
                                        title='Emotions Detected in Code Comments',
                                        color='Emotion'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    # Plot emotions as pie chart
                                    fig = px.pie(
                                        emotions_df,
                                        values='Count',
                                        names='Emotion',
                                        title='Emotion Distribution',
                                        color='Emotion'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Export options
                                st.subheader("Export Data")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # CSV Export
                                    comments_df = pd.DataFrame(processed_comments)
                                    st.markdown(generate_download_link(
                                        comments_df, 
                                        'comments_emotions.csv', 
                                        'Download Comments as CSV'
                                    ), unsafe_allow_html=True)
                                    
                                with col2:
                                    # JSON Export
                                    st.markdown(generate_json_download_link(
                                        processed_comments,
                                        'comments_emotions.json',
                                        'Download Comments as JSON'
                                    ), unsafe_allow_html=True)
                                
                                # Search and filter
                                st.subheader("Comments Browser")
                                
                                # Search
                                search_text = st.text_input("Search in comments", "")
                                
                                # Filter by emotion
                                selected_emotions = st.multiselect(
                                    "Filter by emotion",
                                    list(emotion_counts.keys()),
                                    default=list(emotion_counts.keys())
                                )
                                
                                # Sort options
                                col1, col2 = st.columns(2)
                                with col1:
                                    sort_by = st.selectbox(
                                        "Sort by",
                                        ["file", "line", "top_emotion"]
                                    )
                                with col2:
                                    sort_order = st.selectbox(
                                        "Order",
                                        ["Ascending", "Descending"]
                                    )
                                
                                # Apply filters and sort
                                filtered_comments = [c for c in processed_comments if c.get('top_emotion') in selected_emotions]
                                filtered_comments = filter_data_by_text(filtered_comments, search_text, ['text', 'file'])
                                filtered_comments = sort_data(filtered_comments, sort_by, sort_order)
                                
                                # Pagination
                                results_per_page = st.session_state.settings['results_per_page']
                                total_pages = len(filtered_comments) // results_per_page + (1 if len(filtered_comments) % results_per_page > 0 else 0)
                                
                                if total_pages > 0:
                                    col1, col2, col3 = st.columns([1, 1, 2])
                                    
                                    with col1:
                                        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                                        
                                    with col2:
                                        st.markdown(f"**{len(filtered_comments)}** comments found")
                                        
                                    with col3:
                                        st.markdown(f"Showing page **{page}** of **{total_pages}**")
                                        
                                    start_idx = (page - 1) * results_per_page
                                    end_idx = min(start_idx + results_per_page, len(filtered_comments))
                                    
                                    # Display paginated results
                                    for comment in filtered_comments[start_idx:end_idx]:
                                        with st.expander(f"{comment.get('top_emotion', 'Unknown').capitalize()}: {comment.get('text', '')[:50]}...", expanded=False):
                                            st.markdown(f"**Emotion**: {comment.get('top_emotion', 'Unknown')}")
                                            st.markdown(f"**Text**: *{comment.get('text', 'No text')}*")
                                            if 'file' in comment and 'line' in comment:
                                                st.markdown(f"**Location**: `{comment['file']}:{comment['line']}`")
                                            
                                            # Show emotion scores if available
                                            if 'emotion_scores' in comment:
                                                st.markdown("**Emotion Scores**:")
                                                emotion_scores = comment['emotion_scores']
                                                scores_df = pd.DataFrame({
                                                    'Emotion': list(emotion_scores.keys()),
                                                    'Score': list(emotion_scores.values())
                                                })
                                                scores_df = scores_df.sort_values('Score', ascending=False)
                                                st.dataframe(scores_df)
                                else:
                                    st.info("No comments match your search criteria.")
                            else:
                                st.warning("Could not classify emotions in comments. Check emotion classifier implementation.")
                        else:
                            st.error("Emotion classifier not available. Check your implementation.")
            except Exception as e:
                st.error(f"Error processing comments: {e}")
                st.error(traceback.format_exc())
    
    # Show previously processed comments if available
    elif 'processed_comments' in st.session_state:
        st.info("Showing results from previous scan. Click 'Scan Repository for Comments' to refresh.")
        
        processed_comments = st.session_state.processed_comments
        
        # Display results
        emotion_counts = {}
        for comment in processed_comments:
            if 'top_emotion' in comment:
                emotion = comment['top_emotion']
                if emotion not in emotion_counts:
                    emotion_counts[emotion] = 0
                emotion_counts[emotion] += 1
        
        if emotion_counts:
            # Create DataFrame for visualization
            emotions_df = pd.DataFrame({
                'Emotion': list(emotion_counts.keys()),
                'Count': list(emotion_counts.values())
            })
            
            # Create charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Plot emotions as bar chart
                fig = px.bar(
                    emotions_df,
                    x='Emotion',
                    y='Count',
                    title='Emotions Detected in Code Comments',
                    color='Emotion'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Plot emotions as pie chart
                fig = px.pie(
                    emotions_df,
                    values='Count',
                    names='Emotion',
                    title='Emotion Distribution',
                    color='Emotion'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            st.subheader("Export Data")
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV Export
                comments_df = pd.DataFrame(processed_comments)
                st.markdown(generate_download_link(
                    comments_df, 
                    'comments_emotions.csv', 
                    'Download Comments as CSV'
                ), unsafe_allow_html=True)
                
            with col2:
                # JSON Export
                st.markdown(generate_json_download_link(
                    processed_comments,
                    'comments_emotions.json',
                    'Download Comments as JSON'
                ), unsafe_allow_html=True)
            
            # Search and filter
            st.subheader("Comments Browser")
            
            # Search
            search_text = st.text_input("Search in comments", "")
            
            # Filter by emotion
            selected_emotions = st.multiselect(
                "Filter by emotion",
                list(emotion_counts.keys()),
                default=list(emotion_counts.keys())
            )
            
            # Sort options
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox(
                    "Sort by",
                    ["file", "line", "top_emotion"]
                )
            with col2:
                sort_order = st.selectbox(
                    "Order",
                    ["Ascending", "Descending"]
                )
            
            # Apply filters and sort
            filtered_comments = [c for c in processed_comments if c.get('top_emotion') in selected_emotions]
            filtered_comments = filter_data_by_text(filtered_comments, search_text, ['text', 'file'])
            filtered_comments = sort_data(filtered_comments, sort_by, sort_order)
            
            # Pagination
            results_per_page = st.session_state.settings['results_per_page']
            total_pages = len(filtered_comments) // results_per_page + (1 if len(filtered_comments) % results_per_page > 0 else 0)
            
            if total_pages > 0:
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                    
                with col2:
                    st.markdown(f"**{len(filtered_comments)}** comments found")
                    
                with col3:
                    st.markdown(f"Showing page **{page}** of **{total_pages}**")
                    
                start_idx = (page - 1) * results_per_page
                end_idx = min(start_idx + results_per_page, len(filtered_comments))
                
                # Display paginated results
                for comment in filtered_comments[start_idx:end_idx]:
                    with st.expander(f"{comment.get('top_emotion', 'Unknown').capitalize()}: {comment.get('text', '')[:50]}...", expanded=False):
                        st.markdown(f"**Emotion**: {comment.get('top_emotion', 'Unknown')}")
                        st.markdown(f"**Text**: *{comment.get('text', 'No text')}*")
                        if 'file' in comment and 'line' in comment:
                            st.markdown(f"**Location**: `{comment['file']}:{comment['line']}`")
                        
                        # Show emotion scores if available
                        if 'emotion_scores' in comment:
                            st.markdown("**Emotion Scores**:")
                            emotion_scores = comment['emotion_scores']
                            scores_df = pd.DataFrame({
                                'Emotion': list(emotion_scores.keys()),
                                'Score': list(emotion_scores.values())
                            })
                            scores_df = scores_df.sort_values('Score', ascending=False)
                            st.dataframe(scores_df)
            else:
                st.info("No comments match your search criteria.")

# Commit Analysis
elif analysis_type == "Commit Messages":
    st.header("Commit Message Emotion Analysis")
    
    # Parameters
    repo_path = st.text_input("Repository Path", value=st.session_state.settings['repo_path'])
    col1, col2 = st.columns(2)
    
    with col1:
        max_commits = st.slider("Maximum Commits to Analyze", 10, 1000, st.session_state.settings['max_commits'])
    
    with col2:
        branch_name = st.text_input("Branch Name", value=st.session_state.settings['branch_name'])
    
    # Scan button
    if st.button("Analyze Commit Messages"):
        with st.spinner("Analyzing commit messages..."):
            try:
                if not commit_extractor_available:
                    st.error("CommitExtractor module is not available. Please check your installation.")
                else:
                    # Extract commits
                    extractor = CommitExtractor(repo_path)
                    commits = extractor.extract_commits(max_count=max_commits, branch=branch_name)
                    
                    if not commits:
                        st.warning(f"No commits found in repository: {repo_path}")
                    else:
                        st.success(f"Found {len(commits)} commits!")
                        
                        # Process with emotion classifier
                        if emotion_classifier:
                            processed_commits = []
                            for commit in commits:
                                if 'message' in commit:
                                    try:
                                        emotion_scores = emotion_classifier.classify(commit['message'])
                                        top_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
                                        processed_commits.append({
                                            **commit,
                                            'emotion_scores': emotion_scores,
                                            'top_emotion': top_emotion
                                        })
                                    except Exception as e:
                                        # Include the commit anyway
                                        processed_commits.append(commit)
                            
                            # Store in session state for persistence
                            st.session_state.processed_commits = processed_commits
                            
                            # Display results
                            emotion_counts = {}
                            for commit in processed_commits:
                                if 'top_emotion' in commit:
                                    emotion = commit['top_emotion']
                                    if emotion not in emotion_counts:
                                        emotion_counts[emotion] = 0
                                    emotion_counts[emotion] += 1
                            
                            if emotion_counts:
                                # Create DataFrame for visualization
                                emotions_df = pd.DataFrame({
                                    'Emotion': list(emotion_counts.keys()),
                                    'Count': list(emotion_counts.values())
                                })
                                
                                # Create charts
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Plot emotions as pie chart
                                    fig = px.pie(
                                        emotions_df,
                                        values='Count',
                                        names='Emotion',
                                        title='Emotions Detected in Commit Messages',
                                        color='Emotion'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    # Plot emotions by author if available
                                    author_emotions = {}
                                    for commit in processed_commits:
                                        if 'author' in commit and 'top_emotion' in commit:
                                            author = commit['author']
                                            emotion = commit['top_emotion']
                                            
                                            if author not in author_emotions:
                                                author_emotions[author] = {}
                                            
                                            if emotion not in author_emotions[author]:
                                                author_emotions[author][emotion] = 0
                                            
                                            author_emotions[author][emotion] += 1
                                    
                                    # Convert to DataFrame for visualization
                                    if author_emotions:
                                        author_data = []
                                        for author, emotions in author_emotions.items():
                                            for emotion, count in emotions.items():
                                                author_data.append({
                                                    'Author': author,
                                                    'Emotion': emotion,
                                                    'Count': count
                                                })
                                        
                                        authors_df = pd.DataFrame(author_data)
                                        
                                        fig = px.bar(
                                            authors_df,
                                            x='Author',
                                            y='Count',
                                            color='Emotion',
                                            title='Emotions by Author',
                                            barmode='stack'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.info("No author information available in commits.")
                                
                                # Create time series data
                                time_data = []
                                for commit in processed_commits:
                                    if 'date' in commit and commit['date'] and 'top_emotion' in commit:
                                        try:
                                            date = datetime.fromisoformat(commit['date'].replace('Z', '+00:00'))
                                            time_data.append({
                                                'date': date.date(),
                                                'emotion': commit['top_emotion'],
                                                'message': commit.get('message', '')[:50] + '...' if len(commit.get('message', '')) > 50 else commit.get('message', ''),
                                                'author': commit.get('author', 'Unknown')
                                            })
                                        except Exception as e:
                                            pass
                                
                                if time_data:
                                    time_df = pd.DataFrame(time_data)
                                    time_df = time_df.sort_values('date')
                                    
                                    # Create emotion timeline
                                    st.subheader("Emotional Timeline")
                                    
                                    # Timeline view options
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        color_by = st.selectbox(
                                            "Color by",
                                            ["emotion", "author"]
                                        )
                                    
                                    with col2:
                                        if color_by == "author":
                                            selected_authors = st.multiselect(
                                                "Filter by author",
                                                list(time_df['author'].unique()),
                                                default=list(time_df['author'].unique())[:5]  # Default to first 5 authors
                                            )
                                            time_df = time_df[time_df['author'].isin(selected_authors)]
                                    
                                    timeline_fig = px.scatter(
                                        time_df,
                                        x='date',
                                        y='emotion',
                                        color=color_by,
                                        hover_data=['message', 'author'],
                                        title='Emotional Timeline of Commits'
                                    )
                                    st.plotly_chart(timeline_fig, use_container_width=True)
                                
                                # Export options
                                st.subheader("Export Data")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # CSV Export
                                    commits_df = pd.DataFrame([{k: str(v) if isinstance(v, dict) else v for k, v in commit.items()} for commit in processed_commits])
                                    st.markdown(generate_download_link(
                                        commits_df, 
                                        'commits_emotions.csv', 
                                        'Download Commits as CSV'
                                    ), unsafe_allow_html=True)
                                    
                                with col2:
                                    # JSON Export
                                    st.markdown(generate_json_download_link(
                                        processed_commits,
                                        'commits_emotions.json',
                                        'Download Commits as JSON'
                                    ), unsafe_allow_html=True)
                                
                                # Search and filter
                                st.subheader("Commits Browser")
                                
                                # Search
                                search_text = st.text_input("Search in commits", "")
                                
                                # Filter by emotion
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    selected_emotions = st.multiselect(
                                        "Filter by emotion",
                                        list(emotion_counts.keys()),
                                        default=list(emotion_counts.keys())
                                    )
                                
                                with col2:
                                    # Filter by author if available
                                    authors = list(set(commit.get('author', 'Unknown') for commit in processed_commits if 'author' in commit))
                                    if authors:
                                        selected_authors = st.multiselect(
                                            "Filter by author",
                                            authors,
                                            default=authors
                                        )
                                
                                # Sort options
                                col1, col2 = st.columns(2)
                                with col1:
                                    sort_by = st.selectbox(
                                        "Sort by",
                                        ["date", "author", "top_emotion"]
                                    )
                                with col2:
                                    sort_order = st.selectbox(
                                        "Order",
                                        ["Descending", "Ascending"]
                                    )
                                
                                # Apply filters and sort
                                filtered_commits = [c for c in processed_commits if c.get('top_emotion') in selected_emotions]
                                
                                if 'selected_authors' in locals():
                                    filtered_commits = [c for c in filtered_commits if c.get('author', 'Unknown') in selected_authors]
                                
                                filtered_commits = filter_data_by_text(filtered_commits, search_text, ['message'])
                                filtered_commits = sorted(
                                    filtered_commits, 
                                    key=lambda x: x.get(sort_by, ""), 
                                    reverse=(sort_order == "Descending")
                                )
                                
                                # Pagination
                                results_per_page = st.session_state.settings['results_per_page']
                                total_pages = len(filtered_commits) // results_per_page + (1 if len(filtered_commits) % results_per_page > 0 else 0)
                                
                                if total_pages > 0:
                                    col1, col2, col3 = st.columns([1, 1, 2])
                                    
                                    with col1:
                                        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                                        
                                    with col2:
                                        st.markdown(f"**{len(filtered_commits)}** commits found")
                                        
                                    with col3:
                                        st.markdown(f"Showing page **{page}** of **{total_pages}**")
                                        
                                    start_idx = (page - 1) * results_per_page
                                    end_idx = min(start_idx + results_per_page, len(filtered_commits))
                                    
                                    # Display paginated results
                                    for commit in filtered_commits[start_idx:end_idx]:
                                        message_preview = commit.get('message', '').split('\n')[0][:50]
                                        with st.expander(f"{commit.get('top_emotion', 'Unknown').capitalize()}: {message_preview}...", expanded=False):
                                            st.markdown(f"**Emotion**: {commit.get('top_emotion', 'Unknown')}")
                                            st.markdown(f"**Message**: *{commit.get('message', 'No message')}*")
                                            st.markdown(f"**Author**: {commit.get('author', 'Unknown')}")
                                            st.markdown(f"**Date**: {commit.get('date', 'Unknown')}")
                                            if 'hash' in commit:
                                                st.markdown(f"**Commit**: `{commit['hash'][:7]}`")
                                            
                                            # Show emotion scores if available
                                            if 'emotion_scores' in commit:
                                                st.markdown("**Emotion Scores**:")
                                                emotion_scores = commit['emotion_scores']
                                                scores_df = pd.DataFrame({
                                                    'Emotion': list(emotion_scores.keys()),
                                                    'Score': list(emotion_scores.values())
                                                })
                                                scores_df = scores_df.sort_values('Score', ascending=False)
                                                st.dataframe(scores_df)
                                else:
                                    st.info("No commits match your search criteria.")
                            else:
                                st.warning("Could not classify emotions in commits. Check emotion classifier implementation.")
                        else:
                            st.error("Emotion classifier not available. Check your implementation.")
            except Exception as e:
                st.error(f"Error processing commits: {e}")
                st.error(traceback.format_exc())
    
    # Show previously processed commits if available
    elif 'processed_commits' in st.session_state:
        st.info("Showing results from previous scan. Click 'Analyze Commit Messages' to refresh.")
        
        processed_commits = st.session_state.processed_commits
        
        # Display results
        emotion_counts = {}
        for commit in processed_commits:
            if 'top_emotion' in commit:
                emotion = commit['top_emotion']
                if emotion not in emotion_counts:
                    emotion_counts[emotion] = 0
                emotion_counts[emotion] += 1
        
        if emotion_counts:
            # Create DataFrame for visualization
            emotions_df = pd.DataFrame({
                'Emotion': list(emotion_counts.keys()),
                'Count': list(emotion_counts.values())
            })
            
            # Create charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Plot emotions as pie chart
                fig = px.pie(
                    emotions_df,
                    values='Count',
                    names='Emotion',
                    title='Emotions Detected in Commit Messages',
                    color='Emotion'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Plot emotions by author if available
                author_emotions = {}
                for commit in processed_commits:
                    if 'author' in commit and 'top_emotion' in commit:
                        author = commit['author']
                        emotion = commit['top_emotion']
                        
                        if author not in author_emotions:
                            author_emotions[author] = {}
                        
                        if emotion not in author_emotions[author]:
                            author_emotions[author][emotion] = 0
                        
                        author_emotions[author][emotion] += 1
                
                # Convert to DataFrame for visualization
                if author_emotions:
                    author_data = []
                    for author, emotions in author_emotions.items():
                        for emotion, count in emotions.items():
                            author_data.append({
                                'Author': author,
                                'Emotion': emotion,
                                'Count': count
                            })
                    
                    authors_df = pd.DataFrame(author_data)
                    
                    fig = px.bar(
                        authors_df,
                        x='Author',
                        y='Count',
                        color='Emotion',
                        title='Emotions by Author',
                        barmode='stack'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No author information available in commits.")
            
            # Create time series data
            time_data = []
            for commit in processed_commits:
                if 'date' in commit and commit['date'] and 'top_emotion' in commit:
                    try:
                        date = datetime.fromisoformat(commit['date'].replace('Z', '+00:00'))
                        time_data.append({
                            'date': date.date(),
                            'emotion': commit['top_emotion'],
                            'message': commit.get('message', '')[:50] + '...' if len(commit.get('message', '')) > 50 else commit.get('message', ''),
                            'author': commit.get('author', 'Unknown')
                        })
                    except Exception as e:
                        pass
            
            if time_data:
                time_df = pd.DataFrame(time_data)
                time_df = time_df.sort_values('date')
                
                # Create emotion timeline
                st.subheader("Emotional Timeline")
                
                # Timeline view options
                col1, col2 = st.columns(2)
                with col1:
                    color_by = st.selectbox(
                        "Color by",
                        ["emotion", "author"]
                    )
                
                with col2:
                    if color_by == "author":
                        selected_authors = st.multiselect(
                            "Filter by author",
                            list(time_df['author'].unique()),
                            default=list(time_df['author'].unique())[:5]  # Default to first 5 authors
                        )
                        time_df = time_df[time_df['author'].isin(selected_authors)]
                
                timeline_fig = px.scatter(
                    time_df,
                    x='date',
                    y='emotion',
                    color=color_by,
                    hover_data=['message', 'author'],
                    title='Emotional Timeline of Commits'
                )
                st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Export options
            st.subheader("Export Data")
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV Export
                commits_df = pd.DataFrame([{k: str(v) if isinstance(v, dict) else v for k, v in commit.items()} for commit in processed_commits])
                st.markdown(generate_download_link(
                    commits_df, 
                    'commits_emotions.csv', 
                    'Download Commits as CSV'
                ), unsafe_allow_html=True)
                
            with col2:
                # JSON Export
                st.markdown(generate_json_download_link(
                    processed_commits,
                    'commits_emotions.json',
                    'Download Commits as JSON'
                ), unsafe_allow_html=True)
            
            # Search and filter
            st.subheader("Commits Browser")
            
            # Search
            search_text = st.text_input("Search in commits", "")
            
            # Filter by emotion
            col1, col2 = st.columns(2)
            
            with col1:
                selected_emotions = st.multiselect(
                    "Filter by emotion",
                    list(emotion_counts.keys()),
                    default=list(emotion_counts.keys())
                )
            
            with col2:
                # Filter by author if available
                authors = list(set(commit.get('author', 'Unknown') for commit in processed_commits if 'author' in commit))
                if authors:
                    selected_authors = st.multiselect(
                        "Filter by author",
                        authors,
                        default=authors
                    )
            
            # Sort options
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox(
                    "Sort by",
                    ["date", "author", "top_emotion"]
                )
            with col2:
                sort_order = st.selectbox(
                    "Order",
                    ["Descending", "Ascending"]
                )
            
            # Apply filters and sort
            filtered_commits = [c for c in processed_commits if c.get('top_emotion') in selected_emotions]
            
            if 'selected_authors' in locals():
                filtered_commits = [c for c in filtered_commits if c.get('author', 'Unknown') in selected_authors]
            
            filtered_commits = filter_data_by_text(filtered_commits, search_text, ['message'])
            filtered_commits = sorted(
                filtered_commits, 
                key=lambda x: x.get(sort_by, ""), 
                reverse=(sort_order == "Descending")
            )
            
            # Pagination
            results_per_page = st.session_state.settings['results_per_page']
            total_pages = len(filtered_commits) // results_per_page + (1 if len(filtered_commits) % results_per_page > 0 else 0)
            
            if total_pages > 0:
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                    
                with col2:
                    st.markdown(f"**{len(filtered_commits)}** commits found")
                    
                with col3:
                    st.markdown(f"Showing page **{page}** of **{total_pages}**")
                    
                start_idx = (page - 1) * results_per_page
                end_idx = min(start_idx + results_per_page, len(filtered_commits))
                
                # Display paginated results
                for commit in filtered_commits[start_idx:end_idx]:
                    message_preview = commit.get('message', '').split('\n')[0][:50]
                    with st.expander(f"{commit.get('top_emotion', 'Unknown').capitalize()}: {message_preview}...", expanded=False):
                        st.markdown(f"**Emotion**: {commit.get('top_emotion', 'Unknown')}")
                        st.markdown(f"**Message**: *{commit.get('message', 'No message')}*")
                        st.markdown(f"**Author**: {commit.get('author', 'Unknown')}")
                        st.markdown(f"**Date**: {commit.get('date', 'Unknown')}")
                        if 'hash' in commit:
                            st.markdown(f"**Commit**: `{commit['hash'][:7]}`")
                        
                        # Show emotion scores if available
                        if 'emotion_scores' in commit:
                            st.markdown("**Emotion Scores**:")
                            emotion_scores = commit['emotion_scores']
                            scores_df = pd.DataFrame({
                                'Emotion': list(emotion_scores.keys()),
                                'Score': list(emotion_scores.values())
                            })
                            scores_df = scores_df.sort_values('Score', ascending=False)
                            st.dataframe(scores_df)
            else:
                st.info("No commits match your search criteria.")

# Combined Analysis
elif analysis_type == "Combined Analysis":
    st.header("Combined Code & Commit Analysis")
    
    # Parameters
    repo_path = st.text_input("Repository Path", value=st.session_state.settings['repo_path'])
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_commits = st.slider("Maximum Commits to Analyze", 10, 1000, st.session_state.settings['max_commits'])
    
    with col2:
        branch_name = st.text_input("Branch Name", value=st.session_state.settings['branch_name'])
        
    with col3:
        analyze_options = st.multiselect(
            "What to Analyze",
            ["Comments", "Commits", "Health Metrics", "Emotional Hotspots"],
            default=["Comments", "Commits", "Health Metrics", "Emotional Hotspots"]
        )
    
    # Analyze button
    if st.button("Run Full Repository Analysis"):
        with st.spinner("Analyzing repository..."):
            try:
                # Track what data was successfully analyzed
                analyzed_data = {
                    'comments': False,
                    'commits': False
                }
                
                # Results containers
                comments = []
                commits = []
                
                # Analyze comments if selected
                if "Comments" in analyze_options:
                    if comment_extractor_available:
                        # Extract comments
                        extractor = CommentExtractor()
                        comments = extractor.extract_from_directory(repo_path)
                        st.info(f"Found {len(comments)} code comments")
                        analyzed_data['comments'] = len(comments) > 0
                    else:
                        st.warning("CommentExtractor not available. Skipping comment analysis.")
                
                # Analyze commits if selected
                if "Commits" in analyze_options:
                    if commit_extractor_available:
                        # Extract commits
                        extractor = CommitExtractor(repo_path)
                        commits = extractor.extract_commits(max_count=max_commits, branch=branch_name)
                        st.info(f"Found {len(commits)} commits")
                        analyzed_data['commits'] = len(commits) > 0
                    else:
                        st.warning("CommitExtractor not available. Skipping commit analysis.")
                
                if not analyzed_data['comments'] and not analyzed_data['commits']:
                    st.error("No data could be analyzed. Check your repository path and available extractors.")
                else:
                    # Process with emotion classifier
                    if emotion_classifier:
                        # Process comments
                        processed_comments = []
                        if analyzed_data['comments']:
                            for comment in comments:
                                if 'text' in comment:
                                    try:
                                        emotion_scores = emotion_classifier.classify(comment['text'])
                                        top_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
                                        processed_comments.append({
                                            **comment,
                                            'emotion_scores': emotion_scores,
                                            'top_emotion': top_emotion
                                        })
                                    except Exception as e:
                                        # Include the comment without emotion data
                                        processed_comments.append(comment)
                        
                        # Process commits
                        processed_commits = []
                        if analyzed_data['commits']:
                            for commit in commits:
                                if 'message' in commit:
                                    try:
                                        emotion_scores = emotion_classifier.classify(commit['message'])
                                        top_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
                                        processed_commits.append({
                                            **commit,
                                            'emotion_scores': emotion_scores,
                                            'top_emotion': top_emotion
                                        })
                                    except Exception as e:
                                        # Include the commit without emotion data
                                        processed_commits.append(commit)
                        
                        # Store in session state for persistence
                        st.session_state.processed_comments = processed_comments
                        st.session_state.processed_commits = processed_commits
                        
                        # Calculate statistics
                        stats = {
                            'comments_count': len(processed_comments),
                            'commits_count': len(processed_commits),
                            'comments_emotions': {},
                            'commits_emotions': {}
                        }
                        
                        # Count emotions in comments
                        for comment in processed_comments:
                            if 'top_emotion' in comment:
                                emotion = comment['top_emotion']
                                if emotion not in stats['comments_emotions']:
                                    stats['comments_emotions'][emotion] = 0
                                stats['comments_emotions'][emotion] += 1
                        
                        # Count emotions in commits
                        for commit in processed_commits:
                            if 'top_emotion' in commit:
                                emotion = commit['top_emotion']
                                if emotion not in stats['commits_emotions']:
                                    stats['commits_emotions'][emotion] = 0
                                stats['commits_emotions'][emotion] += 1
                        
                        # Display dashboard
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if analyzed_data['comments']:
                                st.metric("Total Comments", stats['comments_count'])
                                
                                # Comment emotions chart
                                if stats['comments_emotions']:
                                    comments_df = pd.DataFrame({
                                        'Emotion': list(stats['comments_emotions'].keys()),
                                        'Count': list(stats['comments_emotions'].values())
                                    })
                                    
                                    fig = px.pie(
                                        comments_df,
                                        values='Count',
                                        names='Emotion',
                                        title='Emotions in Code Comments',
                                        color='Emotion'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Comments were not analyzed.")
                        
                        with col2:
                            if analyzed_data['commits']:
                                st.metric("Total Commits", stats['commits_count'])
                                
                                # Commit emotions chart
                                if stats['commits_emotions']:
                                    commits_df = pd.DataFrame({
                                        'Emotion': list(stats['commits_emotions'].keys()),
                                        'Count': list(stats['commits_emotions'].values())
                                    })
                                    
                                    fig = px.pie(
                                        commits_df,
                                        values='Count',
                                        names='Emotion',
                                        title='Emotions in Commit Messages',
                                        color='Emotion'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Commits were not analyzed.")
                        
                        # Emotional health metrics
                        if "Health Metrics" in analyze_options:
                            st.subheader("Codebase Health Metrics")
                            
                            # Calculate health score
                            positive_emotions = ['joy', 'pride']
                            negative_emotions = ['frustration', 'anger', 'confusion', 'concern', 'urgency']
                            
                            total_pos_count = 0
                            total_neg_count = 0
                            
                            for emotion, count in stats['comments_emotions'].items():
                                if emotion in positive_emotions:
                                    total_pos_count += count
                                elif emotion in negative_emotions:
                                    total_neg_count += count
                            
                            for emotion, count in stats['commits_emotions'].items():
                                if emotion in positive_emotions:
                                    total_pos_count += count
                                elif emotion in negative_emotions:
                                    total_neg_count += count
                            
                            if total_pos_count + total_neg_count > 0:
                                health_score = (total_pos_count / (total_pos_count + total_neg_count)) * 100
                            else:
                                health_score = 50
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Codebase Health Score", 
                                    f"{health_score:.1f}%",
                                    delta=None if health_score > 60 else f"{60 - health_score:.1f}% below recommended"
                                )
                            
                            with col2:
                                # Calculate burnout risk based on frustration and anger
                                frustration_count = stats['comments_emotions'].get('frustration', 0) + stats['commits_emotions'].get('frustration', 0)
                                anger_count = stats['comments_emotions'].get('anger', 0) + stats['commits_emotions'].get('anger', 0)
                                total_count = stats['comments_count'] + stats['commits_count']
                                
                                if total_count > 0:
                                    burnout_risk = ((frustration_count + anger_count) / total_count) * 100
                                    risk_level = "Low" if burnout_risk < 20 else "Medium" if burnout_risk < 40 else "High"
                                else:
                                    risk_level = "Unknown"
                                
                                st.metric("Burnout Risk", risk_level)
                            
                            with col3:
                                # Technical debt indicators
                                tech_debt_count = 0
                                for comment in processed_comments:
                                    text = comment.get('text', '').lower()
                                    if any(marker in text for marker in [
                                        'todo', 'fixme', 'hack', 'workaround', 'temporary', 
                                        'quick fix', 'refactor later', 'technical debt'
                                    ]):
                                        tech_debt_count += 1
                                
                                st.metric("Technical Debt Indicators", tech_debt_count)
                        
                        # Create timeline
                        if analyzed_data['commits']:
                            st.subheader("Emotional Timeline")
                            
                            timeline_data = []
                            
                            # Add commit data to timeline
                            for commit in processed_commits:
                                if 'date' in commit and 'top_emotion' in commit:
                                    try:
                                        date = datetime.fromisoformat(commit['date'].replace('Z', '+00:00'))
                                        timeline_data.append({
                                            'date': date.date(),
                                            'type': 'Commit',
                                            'emotion': commit['top_emotion'],
                                            'content': commit.get('message', '')[:50] + '...' if len(commit.get('message', '')) > 50 else commit.get('message', ''),
                                            'author': commit.get('author', 'Unknown')
                                        })
                                    except:
                                        # Skip commits with invalid dates
                                        pass
                            
                            if timeline_data:
                                time_df = pd.DataFrame(timeline_data)
                                time_df = time_df.sort_values('date')
                                
                                # Timeline view options
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    color_by = st.selectbox(
                                        "Color by",
                                        ["emotion", "author", "type"]
                                    )
                                
                                with col2:
                                    if color_by == "author" and 'author' in time_df:
                                        authors = list(time_df['author'].unique())
                                        selected_authors = st.multiselect(
                                            "Filter by author",
                                            authors,
                                            default=authors[:5]  # Default to first 5 authors
                                        )
                                        time_df = time_df[time_df['author'].isin(selected_authors)]
                                
                                fig = px.scatter(
                                    time_df,
                                    x='date',
                                    y='emotion',
                                    color=color_by,
                                    hover_data=['content', 'type', 'author'],
                                    title='Emotional Timeline of Development'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Show emotional hotspots
                        if "Emotional Hotspots" in analyze_options and analyzed_data['comments']:
                            st.subheader("Emotional Hotspots in Code")
                            
                            # Group comments by file
                            file_emotions = {}
                            for comment in processed_comments:
                                if 'file' in comment and 'top_emotion' in comment:
                                    file = comment['file']
                                    emotion = comment['top_emotion']
                                    
                                    if file not in file_emotions:
                                        file_emotions[file] = {'count': 0, 'emotions': {}}
                                    
                                    file_emotions[file]['count'] += 1
                                    
                                    if emotion not in file_emotions[file]['emotions']:
                                        file_emotions[file]['emotions'][emotion] = 0
                                    
                                    file_emotions[file]['emotions'][emotion] += 1
                            
                            # Calculate negative ratio for each file
                            hotspot_data = []
                            for file, data in file_emotions.items():
                                if data['count'] >= 3:  # Only consider files with at least 3 comments
                                    neg_count = sum(data['emotions'].get(e, 0) for e in negative_emotions)
                                    neg_ratio = neg_count / data['count'] if data['count'] > 0 else 0
                                    
                                    # Include all files but mark hotspots
                                    is_hotspot = neg_ratio > 0.3  # Files with >30% negative emotions
                                    hotspot_data.append({
                                        'file': os.path.basename(file),
                                        'negative_ratio': neg_ratio,
                                        'comment_count': data['count'],
                                        'is_hotspot': is_hotspot
                                    })
                            
                            if hotspot_data:
                                hotspot_df = pd.DataFrame(hotspot_data)
                                hotspot_df = hotspot_df.sort_values('negative_ratio', ascending=False)
                                
                                # Filter options
                                show_all_files = st.checkbox("Show all files (not just hotspots)", value=False)
                                
                                if not show_all_files:
                                    hotspot_df = hotspot_df[hotspot_df['is_hotspot']]
                                
                                if not hotspot_df.empty:
                                    fig = px.bar(
                                        hotspot_df,
                                        x='file',
                                        y='negative_ratio',
                                        color='comment_count',
                                        title='Files with High Negative Emotion Concentration',
                                        labels={
                                            'negative_ratio': 'Negative Emotion Ratio',
                                            'file': 'File',
                                            'comment_count': 'Comment Count'
                                        },
                                        color_continuous_scale='Viridis'
                                    )
                                    
                                    # Add threshold line at 0.3
                                    fig.add_shape(
                                        type="line",
                                        x0=-0.5,
                                        y0=0.3,
                                        x1=len(hotspot_df) - 0.5,
                                        y1=0.3,
                                        line=dict(
                                            color="red",
                                            width=2,
                                            dash="dash",
                                        )
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    st.markdown("⚠️ Files above the red line (>30% negative emotions) may benefit from refactoring or code review.")
                                    
                                    # Hotspot details
                                    st.subheader("Hotspot Details")
                                    
                                    for file_data in hotspot_df.itertuples():
                                        if file_data.is_hotspot or show_all_files:
                                            with st.expander(f"{'🔥 ' if file_data.is_hotspot else ''}{file_data.file} - {file_data.negative_ratio:.1%} negative", expanded=file_data.is_hotspot):
                                                # Find all comments for this file
                                                file_comments = [c for c in processed_comments if os.path.basename(c.get('file', '')) == file_data.file]
                                                
                                                # Group by emotion
                                                emotions_in_file = {}
                                                for comment in file_comments:
                                                    if 'top_emotion' in comment:
                                                        emotion = comment['top_emotion']
                                                        if emotion not in emotions_in_file:
                                                            emotions_in_file[emotion] = []
                                                        emotions_in_file[emotion].append(comment)
                                                
                                                # Show negative emotions first
                                                for emotion in negative_emotions:
                                                    if emotion in emotions_in_file:
                                                        st.markdown(f"**{emotion.capitalize()}** ({len(emotions_in_file[emotion])})")
                                                        for comment in emotions_in_file[emotion][:3]:  # Show up to 3 examples
                                                            st.markdown(f"- *{comment.get('text', '')}* (Line {comment.get('line', 'unknown')})")
                                                
                                                # Then show other emotions
                                                other_emotions = [e for e in emotions_in_file.keys() if e not in negative_emotions]
                                                for emotion in other_emotions:
                                                    st.markdown(f"**{emotion.capitalize()}** ({len(emotions_in_file[emotion])})")
                                                    for comment in emotions_in_file[emotion][:2]:  # Show up to 2 examples
                                                        st.markdown(f"- *{comment.get('text', '')}* (Line {comment.get('line', 'unknown')})")
                                else:
                                    st.info("No significant emotional hotspots detected in the codebase.")
                            else:
                                st.info("No files with sufficient comments to analyze for hotspots.")
                        
                        # Export options
                        st.subheader("Export Data")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if analyzed_data['comments']:
                                # CSV Export for comments
                                comments_df = pd.DataFrame([{k: str(v) if isinstance(v, dict) else v for k, v in comment.items()} for comment in processed_comments])
                                st.markdown(generate_download_link(
                                    comments_df, 
                                    'comments_emotions.csv', 
                                    'Download Comments as CSV'
                                ), unsafe_allow_html=True)
                                
                                # JSON Export for comments
                                st.markdown(generate_json_download_link(
                                    processed_comments,
                                    'comments_emotions.json',
                                    'Download Comments as JSON'
                                ), unsafe_allow_html=True)
                        
                        with col2:
                            if analyzed_data['commits']:
                                # CSV Export for commits
                                commits_df = pd.DataFrame([{k: str(v) if isinstance(v, dict) else v for k, v in commit.items()} for commit in processed_commits])
                                st.markdown(generate_download_link(
                                    commits_df, 
                                    'commits_emotions.csv', 
                                    'Download Commits as CSV'
                                ), unsafe_allow_html=True)
                                
                                # JSON Export for commits
                                st.markdown(generate_json_download_link(
                                    processed_commits,
                                    'commits_emotions.json',
                                    'Download Commits as JSON'
                                ), unsafe_allow_html=True)
                    else:
                        st.error("Emotion classifier not available. Check your implementation.")
            except Exception as e:
                st.error(f"Error in combined analysis: {e}")
                st.error(traceback.format_exc())

# Developer Profiles
elif analysis_type == "Developer Profiles":
    st.header("Developer Emotional Profiles")
    
    # Parameters
    repo_path = st.text_input("Repository Path", value=st.session_state.settings['repo_path'])
    col1, col2 = st.columns(2)
    
    with col1:
        max_commits = st.slider("Maximum Commits to Analyze", 10, 1000, st.session_state.settings['max_commits'])
    
    with col2:
        branch_name = st.text_input("Branch Name", value=st.session_state.settings['branch_name'])
    
    # Analyze button
    if st.button("Analyze Developer Profiles"):
        with st.spinner("Building developer profiles..."):
            try:
                if not commit_extractor_available:
                    st.error("CommitExtractor module is not available. Please check your installation.")
                else:
                    # Extract commits
                    extractor = CommitExtractor(repo_path)
                    commits = extractor.extract_commits(max_count=max_commits, branch=branch_name)
                    
                    if not commits:
                        st.warning(f"No commits found in repository: {repo_path}")
                    else:
                        st.success(f"Found {len(commits)} commits!")
                        
                        # Process with emotion classifier
                        if emotion_classifier:
                            # Create developer profiles
                            dev_data = {}
                            
                            for commit in commits:
                                if 'author_email' in commit and 'message' in commit:
                                    email = commit['author_email']
                                    name = commit.get('author', email.split('@')[0])
                                    
                                    if email not in dev_data:
                                        dev_data[email] = {
                                            'name': name,
                                            'email': email,
                                            'commit_count': 0,
                                            'emotions': {},
                                            'commits': []
                                        }
                                    
                                    # Add commit count
                                    dev_data[email]['commit_count'] += 1
                                    
                                    # Get emotions
                                    try:
                                        emotion_scores = emotion_classifier.classify(commit['message'])
                                        top_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
                                        
                                        if top_emotion not in dev_data[email]['emotions']:
                                            dev_data[email]['emotions'][top_emotion] = 0
                                        
                                        dev_data[email]['emotions'][top_emotion] += 1
                                        
                                        # Add to commits list
                                        if 'date' in commit:
                                            try:
                                                date = datetime.fromisoformat(commit['date'].replace('Z', '+00:00'))
                                                
                                                dev_data[email]['commits'].append({
                                                    'date': date,
                                                    'message': commit['message'],
                                                    'emotion': top_emotion,
                                                    'hash': commit.get('hash', '')
                                                })
                                            except:
                                                # Skip commits with invalid dates
                                                pass
                                    except Exception as e:
                                        st.error(f"Error classifying commit: {e}")
                            
                            # Store in session state for persistence
                            st.session_state.dev_data = dev_data
                            
                            if dev_data:
                                # Create team overview
                                st.subheader("Team Emotional Analysis")
                                
                                # Prepare developer summary data
                                dev_summary = []
                                positive_emotions = ['joy', 'pride']
                                negative_emotions = ['frustration', 'anger', 'exhaustion']
                                
                                for email, data in dev_data.items():
                                    if data['commit_count'] > 0 and data['emotions']:
                                        # Find dominant emotion
                                        dominant_emotion = max(data['emotions'].items(), key=lambda x: x[1])[0]
                                        
                                        # Calculate positivity ratio
                                        pos_count = sum(data['emotions'].get(e, 0) for e in positive_emotions)
                                        neg_count = sum(data['emotions'].get(e, 0) for e in negative_emotions)
                                        
                                        if pos_count + neg_count > 0:
                                            positivity = pos_count / (pos_count + neg_count)
                                        else:
                                            positivity = 0.5
                                        
                                        dev_summary.append({
                                            'name': data['name'],
                                            'email': email,
                                            'commits': data['commit_count'],
                                            'dominant_emotion': dominant_emotion,
                                            'positivity': positivity
                                        })
                                
                                if dev_summary:
                                    # Create dataframe
                                    dev_df = pd.DataFrame(dev_summary)
                                    
                                    # Filter options
                                    min_commits = st.slider("Minimum commits for inclusion", 1, 50, 5)
                                    dev_df = dev_df[dev_df['commits'] >= min_commits]
                                    
                                    if not dev_df.empty:
                                        # Plot team overview
                                        fig = px.scatter(
                                            dev_df,
                                            x='commits',
                                            y='positivity',
                                            color='dominant_emotion',
                                            hover_data=['name', 'email'],
                                            title='Developer Emotional Profiles',
                                            labels={
                                                'commits': 'Number of Commits',
                                                'positivity': 'Positivity Ratio',
                                                'dominant_emotion': 'Dominant Emotion'
                                            },
                                            size='commits'
                                        )
                                        
                                        # Add horizontal line at 0.5 (neutral)
                                        fig.add_shape(
                                            type="line",
                                            x0=0,
                                            y0=0.5,
                                            x1=dev_df['commits'].max() * 1.1,
                                            y1=0.5,
                                            line=dict(
                                                color="gray",
                                                width=1,
                                                dash="dash",
                                            )
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Team emotional distribution
                                        st.subheader("Team Emotional Distribution")
                                        
                                        # Aggregate emotions across team
                                        team_emotions = {}
                                        for _, data in dev_data.items():
                                            for emotion, count in data['emotions'].items():
                                                if emotion not in team_emotions:
                                                    team_emotions[emotion] = 0
                                                team_emotions[emotion] += count
                                        
                                        team_emotions_df = pd.DataFrame({
                                            'Emotion': list(team_emotions.keys()),
                                            'Count': list(team_emotions.values())
                                        })
                                        
                                        fig = px.bar(
                                            team_emotions_df,
                                            x='Emotion',
                                            y='Count',
                                            color='Emotion',
                                            title='Team Emotional Distribution'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Individual developer analysis
                                        st.subheader("Individual Developer Analysis")
                                        
                                        # Export options
                                        st.markdown(generate_json_download_link(
                                            dev_data,
                                            'developer_profiles.json',
                                            'Download Developer Profiles'
                                        ), unsafe_allow_html=True)
                                        
                                        # Developer selector
                                        selected_dev = st.selectbox(
                                            "Select Developer",
                                            options=list(dev_data.keys()),
                                            format_func=lambda x: f"{dev_data[x]['name']} ({x})"
                                        )
                                        
                                        if selected_dev:
                                            dev = dev_data[selected_dev]
                                            
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                st.markdown(f"### {dev['name']}")
                                                st.markdown(f"**Email:** {dev['email']}")
                                                st.markdown(f"**Total Commits:** {dev['commit_count']}")
                                                
                                                # Emotion distribution
                                                if dev['emotions']:
                                                    emotions_df = pd.DataFrame({
                                                        'Emotion': list(dev['emotions'].keys()),
                                                        'Count': list(dev['emotions'].values())
                                                    })
                                                    
                                                    fig = px.pie(
                                                        emotions_df,
                                                        values='Count',
                                                        names='Emotion',
                                                        title='Emotional Distribution in Commits'
                                                    )
                                                    st.plotly_chart(fig, use_container_width=True)
                                            
                                            with col2:
                                                # Create timeline for developer
                                                if dev['commits']:
                                                    commits_df = pd.DataFrame([
                                                        {
                                                            'date': c['date'].date(),
                                                            'emotion': c['emotion'],
                                                            'message': c['message'][:50] + '...' if len(c['message']) > 50 else c['message']
                                                        }
                                                        for c in dev['commits']
                                                    ])
                                                    
                                                    commits_df = commits_df.sort_values('date')
                                                    
                                                    fig = px.scatter(
                                                        commits_df,
                                                        x='date',
                                                        y='emotion',
                                                        color='emotion',
                                                        hover_data=['message'],
                                                        title='Emotional Timeline'
                                                    )
                                                    st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Generate insights
                                            st.subheader("Developer Insights")
                                            
                                            if dev['commits']:
                                                # Check emotional patterns
                                                emotions_list = [c['emotion'] for c in sorted(dev['commits'], key=lambda x: x['date'])]
                                                
                                                # Check for burnout risk
                                                burnout_risk = False
                                                streak_threshold = 3
                                                
                                                current_streak = 1
                                                for i in range(1, len(emotions_list)):
                                                    if emotions_list[i] in negative_emotions:
                                                        if emotions_list[i-1] in negative_emotions:
                                                            current_streak += 1
                                                            if current_streak >= streak_threshold:
                                                                burnout_risk = True
                                                                break
                                                        else:
                                                            current_streak = 1
                                                
                                                # Generate insights
                                                insights = []
                                                
                                                # Dominant emotion
                                                if dev['emotions']:
                                                    dominant_emotion = max(dev['emotions'].items(), key=lambda x: x[1])[0]
                                                    
                                                    if dominant_emotion == 'joy':
                                                        insights.append("✨ Shows a positive attitude in their commits, which can boost team morale.")
                                                    elif dominant_emotion == 'pride':
                                                        insights.append("✨ Takes pride in their work, which often correlates with code quality and attention to detail.")
                                                    elif dominant_emotion == 'frustration':
                                                        insights.append("⚠️ Shows signs of frustration in commits, which may indicate challenging tasks or technical debt.")
                                                    elif dominant_emotion == 'anger':
                                                        insights.append("⚠️ Expresses anger in commits, which could signal difficult tasks or potential conflicts.")
                                                    elif dominant_emotion == 'exhaustion':
                                                        insights.append("⚠️ Shows signs of exhaustion, which may indicate burnout risk or overwork.")
                                                
                                                # Burnout risk
                                                if burnout_risk:
                                                    insights.append("🔥 **Burnout Risk Detected**: Multiple consecutive negative emotional commits may indicate increasing stress.")
                                                
                                                # Emotional variability
                                                if len(dev['emotions']) >= 4:
                                                    insights.append("🔄 High emotional variability in commits may indicate a dynamic work environment or changing challenges.")
                                                
                                                # Recent trend
                                                if len(dev['commits']) >= 5:
                                                    recent_emotions = [c['emotion'] for c in sorted(dev['commits'], key=lambda x: x['date'])[-5:]]
                                                    pos_recent = sum(1 for e in recent_emotions if e in positive_emotions)
                                                    neg_recent = sum(1 for e in recent_emotions if e in negative_emotions)
                                                    
                                                    if pos_recent >= 3:
                                                        insights.append("📈 Recent commits show a positive emotional trend.")
                                                    elif neg_recent >= 3:
                                                        insights.append("📉 Recent commits show a negative emotional trend that may need attention.")
                                                
                                                # Display insights
                                                if insights:
                                                    for insight in insights:
                                                        st.markdown(f"- {insight}")
                                                else:
                                                    st.info("Not enough data to generate meaningful insights.")
                                            else:
                                                st.info("Not enough commit data to analyze emotional patterns.")
                                            
                                            # Recent commits
                                            st.subheader("Recent Commits")
                                            recent_commits = sorted(dev['commits'], key=lambda x: x['date'], reverse=True)[:5]
                                            
                                            for commit in recent_commits:
                                                with st.expander(f"{commit['emotion'].capitalize()}: {commit['message'][:50]}...", expanded=False):
                                                    st.markdown(f"**Date**: {commit['date'].strftime('%Y-%m-%d')}")
                                                    st.markdown(f"**Emotion**: {commit['emotion']}")
                                                    st.markdown(f"**Message**: {commit['message']}")
                                                    if 'hash' in commit and commit['hash']:
                                                        st.markdown(f"**Commit**: `{commit['hash'][:7]}`")
                                    else:
                                        st.warning(f"No developers with at least {min_commits} commits found.")
                                else:
                                    st.warning("Could not create developer profiles. Check if commits have author information.")
                            else:
                                st.warning("No developer data could be extracted from commits.")
                        else:
                            st.error("Emotion classifier not available. Check your implementation.")
            except Exception as e:
                st.error(f"Error in developer profile analysis: {e}")
                st.error(traceback.format_exc())
    
    # Show previously analyzed data if available
    elif 'dev_data' in st.session_state:
        st.info("Showing results from previous analysis. Click 'Analyze Developer Profiles' to refresh.")
        
        dev_data = st.session_state.dev_data
        
        if dev_data:
            # Create team overview
            st.subheader("Team Emotional Analysis")
            
            # Prepare developer summary data
            dev_summary = []
            positive_emotions = ['joy', 'pride']
            negative_emotions = ['frustration', 'anger', 'exhaustion']
            
            for email, data in dev_data.items():
                if data['commit_count'] > 0 and data['emotions']:
                    # Find dominant emotion
                    dominant_emotion = max(data['emotions'].items(), key=lambda x: x[1])[0]
                    
                    # Calculate positivity ratio
                    pos_count = sum(data['emotions'].get(e, 0) for e in positive_emotions)
                    neg_count = sum(data['emotions'].get(e, 0) for e in negative_emotions)
                    
                    if pos_count + neg_count > 0:
                        positivity = pos_count / (pos_count + neg_count)
                    else:
                        positivity = 0.5
                    
                    dev_summary.append({
                        'name': data['name'],
                        'email': email,
                        'commits': data['commit_count'],
                        'dominant_emotion': dominant_emotion,
                        'positivity': positivity
                    })
            
            if dev_summary:
                # Create dataframe
                dev_df = pd.DataFrame(dev_summary)
                
                # Filter options
                min_commits = st.slider("Minimum commits for inclusion", 1, 50, 5)
                dev_df = dev_df[dev_df['commits'] >= min_commits]
                
                if not dev_df.empty:
                    # Plot team overview
                    fig = px.scatter(
                        dev_df,
                        x='commits',
                        y='positivity',
                        color='dominant_emotion',
                        hover_data=['name', 'email'],
                        title='Developer Emotional Profiles',
                        labels={
                            'commits': 'Number of Commits',
                            'positivity': 'Positivity Ratio',
                            'dominant_emotion': 'Dominant Emotion'
                        },
                        size='commits'
                    )
                    
                    # Add horizontal line at 0.5 (neutral)
                    fig.add_shape(
                        type="line",
                        x0=0,
                        y0=0.5,
                        x1=dev_df['commits'].max() * 1.1,
                        y1=0.5,
                        line=dict(
                            color="gray",
                            width=1,
                            dash="dash",
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Team emotional distribution
                    st.subheader("Team Emotional Distribution")
                    
                    # Aggregate emotions across team
                    team_emotions = {}
                    for _, data in dev_data.items():
                        for emotion, count in data['emotions'].items():
                            if emotion not in team_emotions:
                                team_emotions[emotion] = 0
                            team_emotions[emotion] += count
                    
                    team_emotions_df = pd.DataFrame({
                        'Emotion': list(team_emotions.keys()),
                        'Count': list(team_emotions.values())
                    })
                    
                    fig = px.bar(
                        team_emotions_df,
                        x='Emotion',
                        y='Count',
                        color='Emotion',
                        title='Team Emotional Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Individual developer analysis
                    st.subheader("Individual Developer Analysis")
                    
                    # Export options
                    st.markdown(generate_json_download_link(
                        dev_data,
                        'developer_profiles.json',
                        'Download Developer Profiles'
                    ), unsafe_allow_html=True)
                    
                    # Developer selector
                    selected_dev = st.selectbox(
                        "Select Developer",
                        options=list(dev_data.keys()),
                        format_func=lambda x: f"{dev_data[x]['name']} ({x})"
                    )
                    
                    if selected_dev:
                        dev = dev_data[selected_dev]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"### {dev['name']}")
                            st.markdown(f"**Email:** {dev['email']}")
                            st.markdown(f"**Total Commits:** {dev['commit_count']}")
                            
                            # Emotion distribution
                            if dev['emotions']:
                                emotions_df = pd.DataFrame({
                                    'Emotion': list(dev['emotions'].keys()),
                                    'Count': list(dev['emotions'].values())
                                })
                                
                                fig = px.pie(
                                    emotions_df,
                                    values='Count',
                                    names='Emotion',
                                    title='Emotional Distribution in Commits'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Create timeline for developer
                            if dev['commits']:
                                commits_df = pd.DataFrame([
                                    {
                                        'date': c['date'].date(),
                                        'emotion': c['emotion'],
                                        'message': c['message'][:50] + '...' if len(c['message']) > 50 else c['message']
                                    }
                                    for c in dev['commits']
                                ])
                                
                                commits_df = commits_df.sort_values('date')
                                
                                fig = px.scatter(
                                    commits_df,
                                    x='date',
                                    y='emotion',
                                    color='emotion',
                                    hover_data=['message'],
                                    title='Emotional Timeline'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Generate insights
                        st.subheader("Developer Insights")
                        
                        if dev['commits']:
                            # Check emotional patterns
                            emotions_list = [c['emotion'] for c in sorted(dev['commits'], key=lambda x: x['date'])]
                            
                            # Check for burnout risk
                            burnout_risk = False
                            streak_threshold = 3
                            
                            current_streak = 1
                            for i in range(1, len(emotions_list)):
                                if emotions_list[i] in negative_emotions:
                                    if emotions_list[i-1] in negative_emotions:
                                        current_streak += 1
                                        if current_streak >= streak_threshold:
                                            burnout_risk = True
                                            break
                                    else:
                                        current_streak = 1
                            
                            # Generate insights
                            insights = []
                            
                            # Dominant emotion
                            if dev['emotions']:
                                dominant_emotion = max(dev['emotions'].items(), key=lambda x: x[1])[0]
                                
                                if dominant_emotion == 'joy':
                                    insights.append("✨ Shows a positive attitude in their commits, which can boost team morale.")
                                elif dominant_emotion == 'pride':
                                    insights.append("✨ Takes pride in their work, which often correlates with code quality and attention to detail.")
                                elif dominant_emotion == 'frustration':
                                    insights.append("⚠️ Shows signs of frustration in commits, which may indicate challenging tasks or technical debt.")
                                elif dominant_emotion == 'anger':
                                    insights.append("⚠️ Expresses anger in commits, which could signal difficult tasks or potential conflicts.")
                                elif dominant_emotion == 'exhaustion':
                                    insights.append("⚠️ Shows signs of exhaustion, which may indicate burnout risk or overwork.")
                            
                            # Burnout risk
                            if burnout_risk:
                                insights.append("🔥 **Burnout Risk Detected**: Multiple consecutive negative emotional commits may indicate increasing stress.")
                            
                            # Emotional variability
                            if len(dev['emotions']) >= 4:
                                insights.append("🔄 High emotional variability in commits may indicate a dynamic work environment or changing challenges.")
                            
                            # Recent trend
                            if len(dev['commits']) >= 5:
                                recent_emotions = [c['emotion'] for c in sorted(dev['commits'], key=lambda x: x['date'])[-5:]]
                                pos_recent = sum(1 for e in recent_emotions if e in positive_emotions)
                                neg_recent = sum(1 for e in recent_emotions if e in negative_emotions)
                                
                                if pos_recent >= 3:
                                    insights.append("📈 Recent commits show a positive emotional trend.")
                                elif neg_recent >= 3:
                                    insights.append("📉 Recent commits show a negative emotional trend that may need attention.")
                            
                            # Display insights
                            if insights:
                                for insight in insights:
                                    st.markdown(f"- {insight}")
                            else:
                                st.info("Not enough data to generate meaningful insights.")
                        else:
                            st.info("Not enough commit data to analyze emotional patterns.")
                        
                        # Recent commits
                        st.subheader("Recent Commits")
                        recent_commits = sorted(dev['commits'], key=lambda x: x['date'], reverse=True)[:5]
                        
                        for commit in recent_commits:
                            with st.expander(f"{commit['emotion'].capitalize()}: {commit['message'][:50]}...", expanded=False):
                                st.markdown(f"**Date**: {commit['date'].strftime('%Y-%m-%d')}")
                                st.markdown(f"**Emotion**: {commit['emotion']}")
                                st.markdown(f"**Message**: {commit['message']}")
                                if 'hash' in commit and commit['hash']:
                                    st.markdown(f"**Commit**: `{commit['hash'][:7]}`")
                else:
                    st.warning(f"No developers with at least {min_commits} commits found.")

# Add a footer
st.markdown("---")
st.markdown("### About BERTCodeAuditor")
st.markdown(
    """
    BERTCodeAuditor uses natural language processing to analyze emotions
    hidden in code comments and commit messages. This tool helps teams identify potential
    burnout, track morale, and discover emotional hotspots in their codebase.
    
    **Note:** This is a proof-of-concept demonstration and should be used responsibly 
    with proper privacy considerations.
    """
)