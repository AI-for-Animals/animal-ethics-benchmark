# Eric Zhou to Arturs Kanepajs, 2025-01-21:
# There were 5597 questions after using this criteria: post title ending with a question mark and with some basic keyword searching, various terms related to animal-welfare. script here:
# https://colab.research.google.com/drive/1V3NJ77u_6U_z3Ed2SAWQLiLGtoeMmA_c?usp=sharing
# later I filtered that down to about 2700 by removing some obviously irrelevant questions based on question phrasing
!pip install convokit

from convokit import Corpus, download
import json
import re

class RedditAdvocacyFilter:
    def __init__(self):
        self.advocacy_keywords = {
            'substring': {
                'animal', 'factory farm', 'vegan', 'vegetarian',
                'chicken', 'livestock', 'dairy', 'poultry', 'humane',
                'slaughter', 'plant-based', 'aquaculture', 'speciesism',
                'speciesist'
            },
            'whole_word': {
                'cow', 'cows', 'pig', 'pigs', 'fish', 'fishes', 'fishing',
                'fishery', 'fisheries', 'cage', 'caged', 'cages', 'meat',
                'meatless', 'eggs'
            }
        }
        self.advocacy_keywords_strict = {
            'substring': {
                'animal advocacy', 'animal welfare', 'animal rights',
                'animal protection', 'animal cruelty', 'animal farm',
                'animal agriculture', 'animal liberation',
                'factory farm', 'vegan', 'vegetarian',
                'chicken', 'livestock', 'dairy', 'poultry',
                'slaughter', 'plant-based', 'aquaculture', 'speciesism',
                'speciesist'
            },
            'whole_word': {
                'cow', 'cows', 'pig', 'pigs', 'fishing',
                'fishery', 'fisheries', 'meat', 'meatless', 'humane'
            }
        }

    def _quick_filter(self, text):
        text_lower = text.strip().lower()
        if not text_lower.endswith('?'):
          return False

        for term in self.advocacy_keywords['substring']:
            if term in text_lower:
                return True
        for term in self.advocacy_keywords['whole_word']:
            if re.search(f'\\b{term}\\b', text_lower):
                return True

        return False
    def _quick_filter_vegan(self, text):
        text_lower = text.strip().lower()
        if not text_lower.endswith('?'):
          return False

        return True

    def filter_reddit_corpus(self, corpus):
        """
        Filter Reddit corpus from ConvoKit
        """
        relevant_data = []

        # Process in batches to manage memory
        for conversation in corpus.iter_conversations():
            question = conversation.meta['title']
            if self._quick_filter(question):
                relevant_data.append({
                    'question': question,
                    'tags': [f"subreddit-{conversation.meta.get('subreddit', '')}"],
                })

            # Free up memory
            if len(relevant_data) > 10000:
                yield relevant_data
                relevant_data = []

        if relevant_data:
            yield relevant_data
    def filter_conversation_json(self, conversations):
        relevant_data = []
        for id in conversations:
            conversation = conversations[id]
            question = conversation['title']
            if self._quick_filter(question):
                relevant_data.append({
                    'question': question,
                    'tags': [f"subreddit-{conversation['subreddit']}"],
                })
        yield relevant_data

def process_reddit_corpus():
    """
    Main function to process Reddit corpus
    """
    # Download and load corpus
    subreddit = 'subreddit-DebateAVegan'
    #conversations_file = f'/root/.convokit/saved-corpora/{subreddit}/conversations.json'
    corpus = Corpus(filename=download(subreddit))
    # with open(conversations_file, 'r') as f:
    #   conversations = json.load(f)

    # Initialize filter
    filter = RedditAdvocacyFilter()

    # Process and save results in chunks
    for i, df_chunk in enumerate(filter.filter_reddit_corpus(corpus)):
        # Save each chunk to avoid memory issues
        with open(f'{subreddit}_{i}.json', 'w') as out_file:
          json.dump(df_chunk, out_file, indent=4)

        # Print summary stats for this chunk
        print(f"\nChunk {i} statistics:")
        print(f"Number of relevant posts/comments: {len(df_chunk)}")

if __name__ == "__main__":
    process_reddit_corpus()
