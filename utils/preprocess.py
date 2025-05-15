
# utils/preprocess.py
import re
import string
from typing import List, Dict, Any
import json

def clean_comment(text: str) -> str:
    """Clean and normalize a code comment for processing."""
    # Remove comment markers
    text = re.sub(r'^[/#*\s]+|[/#*\s]+$', '', text)
    
    # Remove code-specific markers
    text = re.sub(r'(TODO|FIXME|NOTE|HACK|XXX):', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove file paths
    text = re.sub(r'[a-zA-Z]:\\[\\\S|*\S]+', '', text)
    text = re.sub(r'(/[\w.-]+)+', '', text)
    
    # Remove excessive punctuation
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    
    # Remove common code patterns
    text = re.sub(r'[a-zA-Z0-9_]+\([^)]*\)', '', text)  # Function calls
    text = re.sub(r'\{[^}]*\}', '', text)  # Curly braces content
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def batch_process_comments(comments: List[Dict[str, Any]], batch_size: int = 32) -> List[List[Dict[str, Any]]]:
    """Split comments into batches for efficient processing."""
    return [comments[i:i+batch_size] for i in range(0, len(comments), batch_size)]

def extract_variables_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Extract variable names from a code file to analyze naming patterns."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return []
        
    variables = []
    _, ext = os.path.splitext(file_path)
    
    if ext == '.py':
        # Match Python variable assignments
        var_matches = re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*', content)
        for match in var_matches:
            variables.append({
                'name': match.group(1),
                'file': file_path,
                'line': content[:match.start()].count('\n') + 1
            })
    
    elif ext in ['.js', '.ts']:
        # Match JavaScript/TypeScript variable declarations
        var_matches = re.finditer(r'(let|var|const)\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
        for match in var_matches:
            variables.append({
                'name': match.group(2),
                'type': match.group(1),
                'file': file_path,
                'line': content[:match.start()].count('\n') + 1
            })
    
    # Analyze for emotional content in variable names
    for var in variables:
        var_words = re.findall(r'[a-z]+', var['name'].lower())
        
        emotional_words = {
            'negative': ['temp', 'hack', 'ugly', 'bad', 'dirty', 'quick', 'workaround', 'fix'],
            'positive': ['clean', 'elegant', 'proper', 'nice', 'good', 'improved'],
            'urgency': ['urgent', 'asap', 'critical', 'important']
        }
        
        for category, words in emotional_words.items():
            if any(word in var_words for word in words):
                var['emotional_category'] = category
                break
                
    return variables

def combine_repo_stats(comments_data: List[Dict], commits_data: List[Dict]) -> Dict[str, Any]:
    """Combine comment and commit data to generate repo-level statistics."""
    stats = {
        'total_comments': len(comments_data),
        'total_commits': len(commits_data),
        'emotional_distribution': {},
        'time_analysis': {},
        'developer_profiles': {}
    }
    
    # Process comment emotions
    comment_emotions = {}
    for comment in comments_data:
        if 'top_emotion' in comment:
            emotion = comment['top_emotion']
            if emotion not in comment_emotions:
                comment_emotions[emotion] = 0
            comment_emotions[emotion] += 1
    
    stats['emotional_distribution']['comments'] = comment_emotions
    
    # Process commits
    commit_emotions = {}
    for commit in commits_data:
        if 'emotion_scores' in commit:
            emotion = max(commit['emotion_scores'].items(), key=lambda x: x[1])[0]
            if emotion not in commit_emotions:
                commit_emotions[emotion] = 0
            commit_emotions[emotion] += 1
    
    stats['emotional_distribution']['commits'] = commit_emotions
    
    # Create developer profiles
    devs = {}
    
    for commit in commits_data:
        if 'author_email' in commit:
            email = commit['author_email']
            
            if email not in devs:
                devs[email] = {
                    'name': commit.get('author', 'Unknown'),
                    'email': email,
                    'commit_count': 0,
                    'emotions': {}
                }
            
            devs[email]['commit_count'] += 1
            
            if 'emotion_scores' in commit:
                emotion = max(commit['emotion_scores'].items(), key=lambda x: x[1])[0]
                if emotion not in devs[email]['emotions']:
                    devs[email]['emotions'][emotion] = 0
                devs[email]['emotions'][emotion] += 1
    
    stats['developer_profiles'] = devs
    
    return stats
