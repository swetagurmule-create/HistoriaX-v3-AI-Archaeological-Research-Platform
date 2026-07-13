"""
NLP Module for Historical and Archaeological Text Analysis
Processes OCR-extracted text to identify entities, relations, and events
"""

import re
import spacy
from langdetect import detect
from deep_translator import GoogleTranslator
from typing import Dict, List, Tuple, Optional


class HistoricalNLPProcessor:
    """Main NLP processor for historical manuscripts"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the NLP processor with spaCy model"""
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Run: python -m spacy download {model_name}")
            raise
        
        # Historical entity patterns for rule-based enhancement
        self.historical_patterns = {
            'DYNASTY': ['dynasty', 'empire', 'kingdom'],
            'ARTIFACT': ['manuscript', 'temple', 'palace', 'fort'],
            'EVENT': ['war', 'battle', 'migration', 'construction', 'founded', 'established']
        }
        
        # Known historical entities (expandable database)
        self.known_entities = {
            'PERSON': ['krishna', 'rama', 'alexander', 'caesar', 'cleopatra', 'buddha', 
                      'ashoka', 'akbar', 'napoleon', 'hannibal', 'aristotle', 'plato',
                      'pandavas', 'kauravas', 'arjuna', 'karna', 'bhima', 'yudhishthira'],
            'GPE': ['dwarka', 'mathura', 'hastinapura', 'kurukshetra', 'ayodhya', 
                   'rome', 'athens', 'sparta', 'babylon', 'persia', 'egypt', 'india',
                   'alexandria', 'constantinople', 'jerusalem', 'mecca', 'pataliputra',
                   'magadha', 'kalinga'],
            'EVENT': ['mahabharata', 'ramayana', 'trojan war', 'battle of thermopylae',
                     'battle of marathon', 'punic wars', 'kalinga war', 'kurukshetra war'],
            'ORG': ['mauryan empire', 'gupta empire', 'mughal empire', 'roman empire',
                   'persian empire', 'ottoman empire']
        }
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize OCR-extracted text"""
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Basic OCR error corrections (expandable)
        ocr_corrections = {
            r'\bl\b': 'I',  # lowercase l to I
            r'0': 'O',      # zero to O in words
        }
        for pattern, replacement in ocr_corrections.items():
            text = re.sub(pattern, replacement, text)
        
        return text.strip()
    
    def detect_language(self, text: str) -> str:
        """Detect the language of input text"""
        try:
            return detect(text)
        except:
            return "unknown"
    
    def translate_text(self, text: str, source_lang: str, target_lang: str = "en") -> str:
        """Translate text to target language"""
        if source_lang == target_lang or source_lang == "unknown":
            return text
        
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            return translator.translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def extract_entities(self, doc, text) -> List[Dict]:
        """Extract named entities with enhanced historical classification"""
        entities = []
        seen_entities = set()
        
        # First, get entities from spaCy
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'type': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            }
            
            # Enhance with historical context
            text_lower = ent.text.lower()
            for hist_type, keywords in self.historical_patterns.items():
                if any(keyword in text_lower for keyword in keywords):
                    entity_info['historical_type'] = hist_type
                    break
            
            entities.append(entity_info)
            seen_entities.add(ent.text.lower())
        
        # Add custom historical entities by checking capitalized words and known entities
        words = text.split()
        for i, word in enumerate(words):
            word_clean = re.sub(r'[^\w]', '', word)
            word_lower = word_clean.lower()
            
            # Skip if already found
            if word_lower in seen_entities or len(word_clean) < 2:
                continue
            
            # Check against known historical entities
            for entity_type, known_list in self.known_entities.items():
                if word_lower in known_list:
                    entities.append({
                        'text': word_clean,
                        'type': entity_type,
                        'start': 0,
                        'end': len(word_clean),
                        'source': 'custom'
                    })
                    seen_entities.add(word_lower)
                    break
            
            # Also check multi-word entities
            if i < len(words) - 1:
                two_word = f"{word_clean} {re.sub(r'[^\w]', '', words[i+1])}".lower()
                for entity_type, known_list in self.known_entities.items():
                    if two_word in known_list:
                        entities.append({
                            'text': two_word.title(),
                            'type': entity_type,
                            'start': 0,
                            'end': len(two_word),
                            'source': 'custom'
                        })
                        seen_entities.add(two_word)
                        break
        
        return entities

    def extract_relations(self, doc, entities) -> List[Dict]:
        """Extract relationships between entities using dependency parsing"""
        relations = []
        
        # Create entity map from both spaCy and custom entities
        entity_map = {}
        entity_text_map = {}
        
        # Map spaCy entities
        for ent in doc.ents:
            for token in ent:
                entity_map[token.i] = {
                    'text': ent.text,
                    'type': ent.label_,
                    'root': ent.root
                }
            entity_text_map[ent.text.lower()] = {
                'text': ent.text,
                'type': ent.label_
            }
        
        # Map custom entities
        for entity in entities:
            if entity.get('source') == 'custom':
                entity_text_map[entity['text'].lower()] = {
                    'text': entity['text'],
                    'type': entity['type']
                }
        
        # Extract relations using dependency parsing
        for token in doc:
            if token.pos_ == "VERB":
                # Find subject and object
                subject = None
                obj = None
                
                for child in token.children:
                    # Look for subject
                    if child.dep_ in ['nsubj', 'nsubjpass']:
                        if child.i in entity_map:
                            subject = entity_map[child.i]
                        elif child.text.lower() in entity_text_map:
                            subject = entity_text_map[child.text.lower()]
                    
                    # Look for object
                    elif child.dep_ in ['dobj', 'pobj', 'attr']:
                        if child.i in entity_map:
                            obj = entity_map[child.i]
                        elif child.text.lower() in entity_text_map:
                            obj = entity_text_map[child.text.lower()]
                
                # If we found both subject and object, create relation
                if subject and obj:
                    relations.append({
                        'subject': subject['text'],
                        'subject_type': subject['type'],
                        'relation': token.lemma_,
                        'object': obj['text'],
                        'object_type': obj['type'],
                        'confidence': 0.85
                    })
        
        return relations
    
    def detect_events(self, doc, entities) -> List[Dict]:
        """Detect historical events in text"""
        events = []
        
        # Extended event patterns
        event_patterns = {
            'construction': ['establish', 'found', 'build', 'construct', 'create', 'erect'],
            'war': ['war', 'battle', 'fight', 'conquer', 'defeat', 'attack', 'invade', 'fought'],
            'migration': ['migrate', 'move', 'travel', 'journey', 'leave', 'depart'],
            'political_change': ['rule', 'reign', 'govern', 'crown', 'abdicate', 'succeed', 'ruled'],
            'destruction': ['destroy', 'demolish', 'ruin', 'collapse', 'fall']
        }
        
        # Check for known event names in entities
        for entity in entities:
            if entity['type'] == 'EVENT':
                events.append({
                    'event_type': 'historical_event',
                    'event_verb': 'mentioned',
                    'sentence': doc.text,
                    'participants': [],
                    'locations': [],
                    'artifacts': [],
                    'time_references': [],
                    'event_name': entity['text'],
                    'confidence': 0.9
                })
        
        for sent in doc.sents:
            for token in sent:
                # Check if token matches any event pattern
                event_type = None
                for category, verbs in event_patterns.items():
                    if token.lemma_ in verbs:
                        event_type = category
                        break
                
                if event_type:
                    # Extract event context
                    event_info = {
                        'event_type': event_type,
                        'event_verb': token.lemma_,
                        'sentence': sent.text,
                        'participants': [],
                        'locations': [],
                        'artifacts': [],
                        'time_references': [],
                        'confidence': 0.8
                    }
                    
                    # Find related entities in sentence
                    for entity in entities:
                        if entity['type'] in ['PERSON', 'ORG']:
                            event_info['participants'].append(entity['text'])
                        elif entity['type'] in ['GPE', 'LOC', 'FAC']:
                            event_info['locations'].append(entity['text'])
                        elif entity['type'] in ['DATE', 'TIME']:
                            event_info['time_references'].append(entity['text'])
                        elif entity['type'] in ['PRODUCT', 'WORK_OF_ART']:
                            event_info['artifacts'].append(entity['text'])
                    
                    events.append(event_info)
                    break
        
        return events
    
    def process(self, text: str, auto_translate: bool = True) -> Dict:
        """Main processing pipeline"""
        # Step 1: Preprocess
        cleaned_text = self.preprocess_text(text)
        
        # Step 2: Language detection
        detected_lang = self.detect_language(cleaned_text)
        
        # Step 3: Translation if needed
        working_text = cleaned_text
        if auto_translate and detected_lang not in ['en', 'unknown']:
            working_text = self.translate_text(cleaned_text, detected_lang, 'en')
        
        # Step 4: NLP processing
        doc = self.nlp(working_text)
        
        # Step 5: Extract information
        entities = self.extract_entities(doc, working_text)
        relations = self.extract_relations(doc, entities)
        events = self.detect_events(doc, entities)
        
        # Step 6: Structure output
        return {
            'metadata': {
                'original_language': detected_lang,
                'translated': detected_lang not in ['en', 'unknown'] and auto_translate,
                'sentence_count': len(list(doc.sents)),
                'token_count': len(doc),
                'original_text': text,
                'cleaned_text': cleaned_text
            },
            'entities': entities,
            'relations': relations,
            'events': events,
            'processed_text': working_text
        }
    
    def to_knowledge_graph(self, result: Dict) -> Dict:
        """Convert processing result to knowledge graph format"""
        nodes = []
        edges = []
        
        # Add entities as nodes
        entity_ids = {}
        for idx, entity in enumerate(result['entities']):
            node_id = f"entity_{idx}"
            entity_ids[entity['text']] = node_id
            nodes.append({
                'id': node_id,
                'label': entity['text'],
                'type': entity['type'],
                'properties': {
                    'historical_type': entity.get('historical_type', 'unknown')
                }
            })
        
        # Add relations as edges
        for idx, relation in enumerate(result['relations']):
            subject_id = entity_ids.get(relation['subject'])
            object_id = entity_ids.get(relation['object'])
            
            if subject_id and object_id:
                edges.append({
                    'id': f"relation_{idx}",
                    'source': subject_id,
                    'target': object_id,
                    'label': relation['relation'],
                    'confidence': relation.get('confidence', 0.5)
                })
        
        # Add events as nodes with connections
        for idx, event in enumerate(result['events']):
            event_id = f"event_{idx}"
            nodes.append({
                'id': event_id,
                'label': f"{event['event_type']}: {event['event_verb']}",
                'type': 'EVENT',
                'properties': {
                    'sentence': event['sentence'],
                    'confidence': event.get('confidence', 0.5)
                }
            })
            
            # Connect event to participants
            for participant in event['participants']:
                if participant in entity_ids:
                    edges.append({
                        'id': f"event_participant_{idx}_{participant}",
                        'source': entity_ids[participant],
                        'target': event_id,
                        'label': 'participated_in'
                    })
            
            # Connect event to locations
            for location in event['locations']:
                if location in entity_ids:
                    edges.append({
                        'id': f"event_location_{idx}_{location}",
                        'source': event_id,
                        'target': entity_ids[location],
                        'label': 'occurred_at'
                    })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': result['metadata']
        }
