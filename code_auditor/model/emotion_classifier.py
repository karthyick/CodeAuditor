# model/emotion_classifier.py
import torch
torch.classes.__path__ = [] # Add this line
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from typing import List, Dict, Any, Tuple
import os
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EmotionClassifier')

class EmotionClassifier:
    """BERT-based classifier for emotions in code comments and commit messages."""
    
    # Emotion categories for classification
    EMOTIONS = [
        'neutral', 'joy', 'frustration', 'anger', 'pride', 
        'exhaustion', 'confusion', 'urgency', 'concern'
    ]
    
    def __init__(self, model_path: str = None):
        """
        Initialize the emotion classifier.
        
        Args:
            model_path: Path to fine-tuned model (optional - will use pre-trained if None)
        """
        logger.info(f"Initializing EmotionClassifier. Model path: {model_path if model_path else 'None (using default)'}")
        logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        try:
            if model_path and os.path.exists(model_path):
                # Load fine-tuned model
                logger.info(f"Loading fine-tuned model from: {model_path}")
                self.tokenizer = BertTokenizer.from_pretrained(model_path)
                logger.info(f"Tokenizer loaded successfully")
                
                self.model = BertForSequenceClassification.from_pretrained(
                    model_path, 
                    num_labels=len(self.EMOTIONS)
                ).to(self.device)
                logger.info(f"Model loaded successfully with {len(self.EMOTIONS)} emotion categories")
                
            else:
                # Use pre-trained BERT and adapt for our task
                logger.info("Loading pre-trained BERT model (bert-base-uncased)")
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                logger.info("Tokenizer loaded successfully")
                
                self.model = BertForSequenceClassification.from_pretrained(
                    'bert-base-uncased',
                    num_labels=len(self.EMOTIONS)
                ).to(self.device)
                logger.info(f"Model loaded successfully with {len(self.EMOTIONS)} emotion categories")
                print("Using pre-trained model. For better results, fine-tune on code comments data.")
                
            logger.info(f"Vocabulary size: {len(self.tokenizer)}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Falling back to rule-based classifier")
            # Fallback to rule-based classifier if model loading fails
            self.model = None
            self.tokenizer = None
    
    def classify(self, text: str) -> Dict[str, float]:
        """
        Classify the emotion in the given text.
        
        Args:
            text: The text to classify (comment or commit message)
            
        Returns:
            Dictionary of emotion scores
        """
        logger.info(f"Classifying text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        if not self.model or not self.tokenizer:
            logger.warning("Model or tokenizer not available, using rule-based classification")
            # Fallback to rule-based classification
            return self._rule_based_classify(text)
        
        # Clean the text
        processed_text = self._preprocess_text(text)
        logger.info(f"Preprocessed text: '{processed_text[:50]}{'...' if len(processed_text) > 50 else ''}'")
        
        # Tokenize and prepare for model
        logger.info("Tokenizing input")
        inputs = self.tokenizer(
            processed_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        logger.debug(f"Input token IDs: {inputs.input_ids}")
        logger.debug(f"Input shape: {inputs.input_ids.shape}")
        
        # Get model prediction
        logger.info("Running BERT inference")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            logger.debug(f"Raw logits: {logits}")
            
            probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
            logger.debug(f"Probability distribution: {probs}")
        
        # Convert to emotion scores
        emotion_scores = {emotion: float(prob) for emotion, prob in zip(self.EMOTIONS, probs)}
        
        # Print sorted scores for easier reading
        sorted_scores = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        logger.info("Emotion scores (sorted):")
        for emotion, score in sorted_scores:
            logger.info(f"  {emotion}: {score:.4f}")
        
        top_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Top emotion: {top_emotion}")
        
        return emotion_scores
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and prepare text for classification."""
        logger.debug(f"Preprocessing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Remove code-specific syntax
        text = text.replace('TODO:', '').replace('FIXME:', '').replace('NOTE:', '')
        logger.debug("Removed code markers")
        
        # Basic cleaning
        text = text.strip()
        if not text:
            logger.warning("Empty text after preprocessing, defaulting to 'neutral comment'")
            return "neutral comment"
            
        logger.debug(f"Preprocessing complete: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        return text
    
    def _rule_based_classify(self, text: str) -> Dict[str, float]:
        """Simple rule-based classification (fallback)."""
        logger.info("Using rule-based classification")
        text = text.lower()
        logger.debug(f"Lowercased text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Default scores
        scores = {emotion: 0.1 for emotion in self.EMOTIONS}
        scores['neutral'] = 0.6  # Default to neutral
        logger.debug("Default scores initialized")
        
        # Simple keyword matching
        matches = []
        
        if any(word in text for word in ['fix', 'bug', 'issue', 'problem', 'error']):
            scores['frustration'] = 0.4
            scores['neutral'] = 0.3
            matches.append("frustration (fix/bug/issue/problem/error)")
            
        if any(word in text for word in ['urgent', 'asap', 'immediately', 'critical']):
            scores['urgency'] = 0.5
            scores['neutral'] = 0.2
            matches.append("urgency (urgent/asap/immediately/critical)")
            
        if any(word in text for word in ['!', '!!', '???', 'wtf', 'omg']):
            scores['anger'] = 0.4
            scores['neutral'] = 0.2
            matches.append("anger (!/!!/???/wtf/omg)")
            
        if any(word in text for word in ['finally', 'works', 'solved', 'hooray', 'yay']):
            scores['joy'] = 0.5
            scores['pride'] = 0.3
            scores['neutral'] = 0.1
            matches.append("joy/pride (finally/works/solved/hooray/yay)")
            
        if any(word in text for word in ['refactor', 'clean', 'improve', 'optimize']):
            scores['pride'] = 0.4
            scores['neutral'] = 0.3
            matches.append("pride (refactor/clean/improve/optimize)")
            
        if any(word in text for word in ['hack', 'workaround', 'temporary', 'bandaid']):
            scores['concern'] = 0.4
            scores['neutral'] = 0.3
            matches.append("concern (hack/workaround/temporary/bandaid)")
            
        if any(word in text for word in ['not sure', 'maybe', 'might', 'investigate']):
            scores['confusion'] = 0.4
            scores['neutral'] = 0.3
            matches.append("confusion (not sure/maybe/might/investigate)")
            
        if any(word in text for word in ['todo', 'fixme', 'xxx', 'revisit']):
            scores['concern'] = 0.3
            scores['neutral'] = 0.4
            matches.append("concern (todo/fixme/xxx/revisit)")
        
        if matches:
            logger.info(f"Keyword matches found: {', '.join(matches)}")
        else:
            logger.info("No keyword matches found, defaulting to neutral")
        
        # Print sorted scores for easier reading
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        logger.info("Rule-based emotion scores (sorted):")
        for emotion, score in sorted_scores:
            logger.info(f"  {emotion}: {score:.4f}")
        
        top_emotion = max(scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Top emotion (rule-based): {top_emotion}")
            
        return scores
    
    def batch_classify(self, texts: List[str]) -> List[Dict[str, float]]:
        """Classify multiple texts efficiently."""
        logger.info(f"Batch classifying {len(texts)} texts")
        
        if self.model and self.tokenizer and len(texts) > 1:
            logger.info("Using batched BERT inference")
            results = []
            
            # Process in batches of 16
            batch_size = 16
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(batch_texts)} texts)")
                
                # Preprocess and tokenize
                processed_texts = [self._preprocess_text(text) for text in batch_texts]
                inputs = self.tokenizer(
                    processed_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                
                # Convert to emotion scores
                for j, probs_array in enumerate(probs):
                    emotion_scores = {emotion: float(prob) for emotion, prob in zip(self.EMOTIONS, probs_array)}
                    results.append(emotion_scores)
                    
                    # Log results sparingly to avoid clutter
                    if j % 5 == 0:
                        top_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
                        logger.debug(f"Text {i+j}: Top emotion = {top_emotion}")
                        
            logger.info(f"Batch classification complete for {len(texts)} texts")
            return results
        else:
            logger.info("Using sequential classification (model not available or single text)")
            # Fall back to sequential classification
            return [self.classify(text) for text in texts]
    
    def process_comments(self, comments_file: str, output_file: str = None) -> List[Dict[str, Any]]:
        """
        Process a file of code comments and add emotion classification.
        
        Args:
            comments_file: Path to JSON file containing code comments
            output_file: Path to save processed results (optional)
            
        Returns:
            List of comment objects with emotion scores added
        """
        logger.info(f"Processing comments from file: {comments_file}")
        try:
            with open(comments_file, 'r', encoding='utf-8') as f:
                comments = json.load(f)
                logger.info(f"Loaded {len(comments)} comments from file")
        except Exception as e:
            logger.error(f"Error loading comments file: {comments_file}")
            logger.error(str(e))
            return []
            
        processed_comments = []
        
        # Process comments in batches if model is available
        if self.model and self.tokenizer:
            logger.info("Processing comments in batch mode")
            comment_texts = []
            comment_indices = []
            
            # Collect comment texts and indices
            for i, comment in enumerate(comments):
                if 'text' in comment:
                    comment_texts.append(comment['text'])
                    comment_indices.append(i)
            
            # Batch classify
            logger.info(f"Batch classifying {len(comment_texts)} comments")
            emotion_scores_batch = self.batch_classify(comment_texts)
            
            # Match results back to comments
            for i, scores in zip(comment_indices, emotion_scores_batch):
                comment = comments[i]
                top_emotion = max(scores.items(), key=lambda x: x[1])
                
                processed_comment = {
                    **comment,
                    'emotion_scores': scores,
                    'top_emotion': top_emotion[0],
                    'top_emotion_score': top_emotion[1]
                }
                
                processed_comments.append(processed_comment)
                
                # Log sparingly to avoid clutter
                if len(processed_comments) % 20 == 0:
                    logger.info(f"Processed {len(processed_comments)}/{len(comment_indices)} comments")
        else:
            # Process sequentially
            logger.info("Processing comments one by one")
            for i, comment in enumerate(comments):
                if 'text' in comment:
                    # Classify the comment text
                    emotion_scores = self.classify(comment['text'])
                    
                    # Add top emotion
                    top_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                    
                    processed_comment = {
                        **comment,
                        'emotion_scores': emotion_scores,
                        'top_emotion': top_emotion[0],
                        'top_emotion_score': top_emotion[1]
                    }
                    
                    processed_comments.append(processed_comment)
                    
                    # Log progress
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(comments)} comments")
        
        logger.info(f"Completed processing {len(processed_comments)} comments")
        
        # Summarize emotions
        emotion_counts = {}
        for comment in processed_comments:
            emotion = comment.get('top_emotion', 'unknown')
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += 1
        
        logger.info("Emotion distribution in comments:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(processed_comments)) * 100
            logger.info(f"  {emotion}: {count} ({percentage:.1f}%)")
        
        # Save results
        if output_file:
            logger.info(f"Saving processed comments to: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_comments, f, indent=2)
                
        return processed_comments

# Add ability to run as standalone script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify emotions in code comments")
    parser.add_argument("--model", help="Path to fine-tuned model", default=None)
    parser.add_argument("--input", help="Path to input JSON file with comments", required=True)
    parser.add_argument("--output", help="Path to output JSON file", default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    # Run classification
    classifier = EmotionClassifier(model_path=args.model)
    processed_comments = classifier.process_comments(args.input, args.output)
    
    print(f"\nProcessed {len(processed_comments)} comments")
    print("Done!")