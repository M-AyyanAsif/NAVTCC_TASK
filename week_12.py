import os
import json
import re
import math
from datetime import datetime

# ==============================================================================
# DAY 1: PROCESS VIDEOS (Azure Video Indexer & Computer Vision Video Solutions)
# ==============================================================================
class AzureVideoIndexerClient:
    """
    Simulates Azure Video Indexer API pipeline:
    Ingests video files, performs multimodal audio-visual indexing,
    extracts keyframes, transcriptions, faces, OCR, and sentiment timeline.
    """
    def __init__(self, account_id="mock-account-101", location="trial"):
        self.account_id = account_id
        self.location = location

    def index_video(self, video_name, duration_seconds=120):
        """Simulates end-to-end video processing and insights extraction."""
        return {
            "videoId": f"vid_{abs(hash(video_name)) % 100000}",
            "name": video_name,
            "duration": f"{duration_seconds}s",
            "state": "Processed",
            "insights": {
                "transcript": [
                    {"id": 1, "start": "00:00:02", "end": "00:00:15", "text": "Welcome to the NAVTTC AI and Deep Learning training session."},
                    {"id": 2, "start": "00:00:16", "end": "00:00:35", "text": "Today we deploy Azure Cognitive Services and Knowledge Mining pipelines."}
                ],
                "keyframes": [
                    {"frameId": "kf_01", "timestamp": "00:00:05", "label": "Title Slide"},
                    {"frameId": "kf_02", "timestamp": "00:00:20", "label": "Architecture Diagram"},
                    {"frameId": "kf_03", "timestamp": "00:01:10", "label": "Live Code Demo"}
                ],
                "faces": [
                    {"id": 1, "name": "Instructor", "confidence": 0.96, "appearances": ["00:00:02-00:00:45"]}
                ],
                "topics": ["Artificial Intelligence", "Cloud Computing", "Deep Learning", "Azure Cognitive Services"],
                "sentiment_summary": {"positive": 0.85, "neutral": 0.12, "negative": 0.03}
            }
        }


# ==============================================================================
# DAY 2: AZURE NLP, SPEECH & TRANSLATION SOLUTIONS
# ==============================================================================
class AzureLanguageServiceClient:
    """
    Azure AI Language (Text Analytics) Client.
    Covers sentiment analysis, opinion mining, key phrase extraction, and NER.
    """
    def __init__(self, endpoint=None, key=None):
        self.endpoint = endpoint or "https://navttc-lang.cognitiveservices.azure.com"
        self.key = key or "DEMO_KEY"

    def analyze_sentiment(self, text):
        pos_words = {"great", "excellent", "love", "good", "helpful", "fast", "powerful", "clean"}
        neg_words = {"bad", "slow", "error", "horrible", "broken", "worst", "fail"}
        tokens = re.findall(r'\b\w+\b', text.lower())
        pos_count = sum(1 for t in tokens if t in pos_words)
        neg_count = sum(1 for t in tokens if t in neg_words)

        if pos_count > neg_count:
            sentiment = "positive"
            conf = min(0.99, 0.6 + 0.1 * pos_count)
        elif neg_count > pos_count:
            sentiment = "negative"
            conf = min(0.99, 0.6 + 0.1 * neg_count)
        else:
            sentiment = "neutral"
            conf = 0.80

        return {
            "text": text,
            "sentiment": sentiment,
            "confidence_scores": {
                "positive": round(conf if sentiment == "positive" else (0.1 if sentiment == "negative" else 0.1), 2),
                "neutral": round(conf if sentiment == "neutral" else 0.1, 2),
                "negative": round(conf if sentiment == "negative" else 0.05, 2)
            }
        }

    def extract_key_phrases(self, text):
        stop = {"a", "an", "the", "in", "on", "at", "to", "for", "with", "is", "are", "and", "this", "that"}
        words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()) if w not in stop]
        # Return unique significant keywords
        return list(dict.fromkeys(words))[:6]

    def recognize_entities(self, text):
        entities = []
        # Match dates
        dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b', text)
        for d in dates:
            entities.append({"text": d, "category": "DateTime", "confidence": 0.95})
        # Match Organizations
        orgs = ["Microsoft", "Azure", "NAVTTC", "Google", "Amazon", "OpenAI"]
        for org in orgs:
            if re.search(r'\b' + re.escape(org) + r'\b', text, re.IGNORECASE):
                entities.append({"text": org, "category": "Organization", "confidence": 0.98})
        # Match Locations
        locs = ["Pakistan", "Islamabad", "Lahore", "Karachi", "United States", "New York"]
        for loc in locs:
            if re.search(r'\b' + re.escape(loc) + r'\b', text, re.IGNORECASE):
                entities.append({"text": loc, "category": "Location", "confidence": 0.97})
        return entities


class AzureSpeechAndTranslationClient:
    """Simulates Speech-to-Text, Text-to-Speech synthesis, and Multilingual Translation."""

    DICTIONARY_URDU = {
        "hello": "ہیلو",
        "welcome": "خوش آمدید",
        "artificial intelligence": "مصنوعی ذہانت",
        "deep learning": "ڈیپ لرننگ",
        "thank you": "شکریہ",
        "course": "کورس"
    }

    @staticmethod
    def speech_to_text(audio_stream_mock):
        return {
            "status": "RecognitionStatus.Success",
            "recognized_text": "Welcome to NAVTTC Artificial Intelligence course.",
            "duration": "00:00:03.450",
            "confidence": 0.97
        }

    @staticmethod
    def text_to_speech(text, voice_name="en-US-JennyNeural"):
        return {
            "status": "SynthesisCompleted",
            "voice_name": voice_name,
            "synthesized_text": text,
            "audio_format": "audio/16khz-128kbitrate-mono-mp3",
            "audio_size_bytes": len(text) * 320
        }

    @classmethod
    def translate_text(cls, text, to_lang="ur"):
        translated = text.lower()
        for en_phrase, ur_trans in cls.DICTIONARY_URDU.items():
            translated = translated.replace(en_phrase, ur_trans)
        return {
            "source_text": text,
            "target_language": to_lang,
            "translated_text": translated,
            "detected_language": {"language": "en", "score": 1.0}
        }


# ==============================================================================
# DAY 3: LANGUAGE UNDERSTANDING (CLU / LUIS) & QUESTION ANSWERING
# ==============================================================================
class ConversationalLanguageUnderstanding:
    """
    Conversational Language Understanding (CLU / LUIS).
    Extracts intents and named entities from natural language user queries.
    """
    def __init__(self, project_name="NavttcStudentSupport"):
        self.project_name = project_name
        self.intents = {
            "CheckCourseSchedule": [r"schedule", r"timetable", r"when.*class", r"timing"],
            "EnrollInModule": [r"enroll", r"register", r"sign up", r"admission"],
            "CheckGrades": [r"grade", r"marks", r"result", r"score", r"assessment"],
            "GetAssistance": [r"help", r"support", r"contact", r"mentor", r"trainer"]
        }

    def predict_intent_and_entities(self, utterance):
        utterance_clean = utterance.lower()
        predicted_intent = "None"
        max_score = 0.0

        for intent, patterns in self.intents.items():
            matches = sum(1 for p in patterns if re.search(p, utterance_clean))
            if matches > 0:
                score = 0.70 + 0.15 * matches
                if score > max_score:
                    max_score = min(0.99, score)
                    predicted_intent = intent

        # Extract entities like Week numbers, Course names
        entities = []
        week_match = re.search(r'\bweek\s*(\d+)\b', utterance_clean)
        if week_match:
            entities.append({"category": "WeekNumber", "text": week_match.group(0), "value": week_match.group(1)})

        return {
            "utterance": utterance,
            "topIntent": predicted_intent if max_score > 0 else "None",
            "confidence": round(max_score if max_score > 0 else 0.25, 4),
            "entities": entities
        }


class AzureQuestionAnsweringKB:
    """Azure Question Answering Knowledge Base (formerly QnA Maker)."""
    def __init__(self):
        self.knowledge_base = [
            {
                "questions": ["How long is this AI course?", "What is the course duration?", "Duration of program"],
                "answer": "The AI (Machine Learning & Deep Learning) course duration is 3 Months (4 hours/day)."
            },
            {
                "questions": ["What is the breakdown of theory versus practicals?", "Practical percentage"],
                "answer": "The course execution plan is 20% Theory and 80% Practical Hands-on."
            },
            {
                "questions": ["Which companies offer jobs after this program?", "Placement companies"],
                "answer": "Companies include Careem, Afiniti, Addo.ai, Arbisoft, i2c, Xavor, FiveRivers, Confiz, NetSol, and Research Institutes."
            }
        ]

    def query(self, user_question):
        user_words = set(re.findall(r'\b\w+\b', user_question.lower()))
        best_match = None
        highest_overlap = 0

        for item in self.knowledge_base:
            for q in item["questions"]:
                q_words = set(re.findall(r'\b\w+\b', q.lower()))
                overlap = len(user_words.intersection(q_words))
                if overlap > highest_overlap:
                    highest_overlap = overlap
                    best_match = item["answer"]

        confidence = min(0.98, 0.4 + 0.2 * highest_overlap) if highest_overlap > 0 else 0.15
        return {
            "query": user_question,
            "answer": best_match if best_match else "I'm sorry, I could not find a relevant answer in the knowledge base.",
            "confidence": round(confidence, 2)
        }


# ==============================================================================
# DAY 4: AZURE KNOWLEDGE MINING SOLUTIONS (Azure AI Search & Skillsets)
# ==============================================================================
class AzureKnowledgeMiningPipeline:
    """
    Implements an end-to-end Azure AI Search (Cognitive Search) Pipeline:
      1. Data Source (Unstructured documents / PDF files)
      2. Cognitive Skillset (OCR skill, Key Phrase Extraction, Entity Recognition)
      3. Search Index Schema (Searchable, filterable, facetable fields)
      4. Search Indexer Execution & Semantic Hybrid Querying
    """
    def __init__(self, index_name="navttc-course-docs"):
        self.index_name = index_name
        self.index_schema = {
            "name": index_name,
            "fields": [
                {"name": "doc_id", "type": "Edm.String", "key": True},
                {"name": "content", "type": "Edm.String", "searchable": True},
                {"name": "key_phrases", "type": "Collection(Edm.String)", "facetable": True},
                {"name": "entities", "type": "Collection(Edm.String)", "facetable": True},
                {"name": "category", "type": "Edm.String", "filterable": True}
            ]
        }
        self.documents_store = []

    def run_enrichment_skillset(self, doc_id, raw_content, category):
        """Cognitive Skillset Execution: Enriches raw document using AI skills."""
        lang_client = AzureLanguageServiceClient()
        key_phrases = lang_client.extract_key_phrases(raw_content)
        entities = [e["text"] for e in lang_client.recognize_entities(raw_content)]

        enriched_doc = {
            "doc_id": doc_id,
            "content": raw_content,
            "key_phrases": key_phrases,
            "entities": entities,
            "category": category,
            "indexed_at": datetime.now().isoformat()
        }
        self.documents_store.append(enriched_doc)
        return enriched_doc

    def search(self, search_text, filter_category=None):
        """Simulates Azure AI Search semantic matching over enriched index."""
        search_terms = set(re.findall(r'\b\w+\b', search_text.lower()))
        results = []

        for doc in self.documents_store:
            if filter_category and doc["category"] != filter_category:
                continue
            doc_words = set(re.findall(r'\b\w+\b', doc["content"].lower()))
            overlap = len(search_terms.intersection(doc_words))
            if overlap > 0:
                score = round(overlap / len(search_terms), 3)
                results.append({"doc_id": doc["doc_id"], "score": score, "document": doc})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# ==============================================================================
# DAY 5: AZURE CONVERSATIONAL AI SOLUTIONS (Bot Framework & Dialogs)
# ==============================================================================
class AzureConversationalBot:
    """
    Azure Bot Service enterprise conversation engine.
    Orchestrates user turns between CLU intent routing and Question Answering.
    """
    def __init__(self):
        self.clu = ConversationalLanguageUnderstanding()
        self.qa_kb = AzureQuestionAnsweringKB()
        self.conversation_history = []

    def on_message_turn(self, user_message, user_id="user_1"):
        # 1. Check Language Understanding
        intent_result = self.clu.predict_intent_and_entities(user_message)

        if intent_result["topIntent"] == "CheckCourseSchedule":
            response = "Classes are conducted 4 hours per day, 5 days a week. Practical hands-on makes up 80%."
        elif intent_result["topIntent"] == "EnrollInModule":
            response = "Enrollment is managed under the Prime Minister's Hunarmand Pakistan Program (NAVTTC)."
        elif intent_result["topIntent"] == "CheckGrades":
            response = "Assessment records and quizzes are reviewed at the end of each module."
        else:
            # Fallback to Question Answering Knowledge Base
            qa_result = self.qa_kb.query(user_message)
            response = qa_result["answer"]

        turn_log = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "user_message": user_message,
            "detected_intent": intent_result["topIntent"],
            "bot_reply": response
        }
        self.conversation_history.append(turn_log)

        # Generate Adaptive Card payload format
        adaptive_card = {
            "type": "AdaptiveCard",
            "version": "1.3",
            "body": [
                {"type": "TextBlock", "text": "NAVTTC Virtual Assistant", "weight": "Bolder", "size": "Medium"},
                {"type": "TextBlock", "text": response, "wrap": True}
            ]
        }

        return {
            "text": response,
            "intent": intent_result["topIntent"],
            "adaptive_card": adaptive_card
        }


# ==============================================================================
# MAIN TEST & DEMONSTRATION SUITE
# ==============================================================================
def run_all_week12_tasks():
    print("=" * 80)
    print("RUNNING WEEK 12 TASKS: AZURE NLP, SPEECH, KNOWLEDGE MINING & BOTS")
    print("=" * 80)

    # 1. Day 1: Process Videos
    print("\n[Day 1] Process Videos with Azure Video Indexer:")
    video_client = AzureVideoIndexerClient()
    video_result = video_client.index_video("AI_Lecture_Week12.mp4", duration_seconds=120)
    print(f"  Processed Video: {video_result['name']} (ID: {video_result['videoId']})")
    print(f"  Detected Keyframes: {[kf['label'] for kf in video_result['insights']['keyframes']]}")
    print(f"  Extracted Topics:   {video_result['insights']['topics']}")
    print(f"  Transcript Sample:  \"{video_result['insights']['transcript'][0]['text']}\"")

    # 2. Day 2: Azure NLP, Speech & Translation
    print("\n[Day 2] Azure NLP Solutions (Text Analytics, Speech & Translation):")
    lang_client = AzureLanguageServiceClient()
    test_review = "The Microsoft Azure machine learning tools are very powerful, fast, and helpful!"
    sentiment = lang_client.analyze_sentiment(test_review)
    key_phrases = lang_client.extract_key_phrases(test_review)
    entities = lang_client.recognize_entities("NAVTTC training was organized in Islamabad, Pakistan with Microsoft on Dec 15, 2026.")
    print(f"  Text Sentiment: {sentiment['sentiment'].upper()} (Confidence: {sentiment['confidence_scores']['positive']})")
    print(f"  Extracted Key Phrases: {key_phrases}")
    print(f"  Recognized Entities:   {[(e['text'], e['category']) for e in entities]}")

    # Speech & Translation
    trans = AzureSpeechAndTranslationClient.translate_text("Hello and welcome to the artificial intelligence course!", to_lang="ur")
    tts = AzureSpeechAndTranslationClient.text_to_speech("Welcome to the Azure Deep Learning practical.")
    print(f"  Translation (English -> Urdu): '{trans['translated_text']}'")
    print(f"  Text-to-Speech Synthesis: Voice={tts['voice_name']}, Status={tts['status']}")

    # 3. Day 3: Language Understanding (CLU) & Question Answering
    print("\n[Day 3] Conversational Language Understanding (CLU) & Question Answering:")
    clu = ConversationalLanguageUnderstanding()
    user_query = "What is the schedule and timetable for week 12 classes?"
    clu_res = clu.predict_intent_and_entities(user_query)
    print(f"  User Utterance: '{user_query}'")
    print(f"  Predicted Top Intent: {clu_res['topIntent']} (Confidence: {clu_res['confidence']})")
    print(f"  Extracted Entities:   {clu_res['entities']}")

    qa = AzureQuestionAnsweringKB()
    qa_res = qa.query("What is the course duration and schedule?")
    print(f"  QA Knowledge Base Answer: \"{qa_res['answer']}\"")

    # 4. Day 4: Azure Knowledge Mining (Cognitive Search)
    print("\n[Day 4] Azure Knowledge Mining Solutions (Azure AI Search Pipeline):")
    km = AzureKnowledgeMiningPipeline("navttc-course-docs")
    km.run_enrichment_skillset(
        doc_id="doc_01",
        raw_content="NAVTTC syllabus includes Machine Learning, Deep Learning, and Microsoft Azure Cognitive Services in Islamabad.",
        category="Curriculum"
    )
    km.run_enrichment_skillset(
        doc_id="doc_02",
        raw_content="Students practice Neural Networks, Convolutional Networks, and Bidirectional LSTM models for natural language processing.",
        category="DeepLearning"
    )

    search_query = "Cognitive Services Microsoft"
    search_hits = km.search(search_query)
    print(f"  Search Query: '{search_query}'")
    print(f"  Hits Found: {len(search_hits)}")
    for hit in search_hits:
        print(f"    Match Doc ID: {hit['doc_id']} | Score: {hit['score']}")
        print(f"    Key Phrases:  {hit['document']['key_phrases']}")

    # 5. Day 5: Conversational AI (Bot Service & Dialog Management)
    print("\n[Day 5] Azure Conversational AI Solutions (Enterprise Bot Framework):")
    bot = AzureConversationalBot()
    bot_turn_1 = bot.on_message_turn("Can you tell me the timetable and schedule for class?")
    print(f"  User: 'Can you tell me the timetable and schedule for class?'")
    print(f"  Bot:  \"{bot_turn_1['text']}\" (Intent: {bot_turn_1['intent']})")

    bot_turn_2 = bot.on_message_turn("What is the percentage of practicals?")
    print(f"  User: 'What is the percentage of practicals?'")
    print(f"  Bot:  \"{bot_turn_2['text']}\"")

    print("=" * 80)
    print("WEEK 12 ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_week12_tasks()
