from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import json
import os
from openai import OpenAI


@dataclass
class Memory:
    timestamp: datetime
    context: str
    location: str
    participants: List[str]
    emotions: List[str]
    sensory_details: Dict[str, str]
    significance: int  # 1-10
    related_memories: List[str]
    tags: List[str]


@dataclass
class Document:
    title: str
    content: str
    type: str  # email, note, report, diary, etc.
    created_at: datetime
    author: str
    metadata: Dict[str, any]
    associated_memories: List[str]
    version_history: List[Dict[str, any]]


class MemoryGraph:
    """Manages interconnected memories and their relationships"""

    def __init__(self):
        self.memories: Dict[str, Memory] = {}
        self.connections: Dict[str, List[str]] = {}
        self.emotional_index: Dict[str, List[str]] = {}
        self.temporal_index: Dict[datetime, List[str]] = {}

    def add_memory(self, memory_id: str, memory: Memory):
        """Add a memory and establish its connections"""
        self.memories[memory_id] = memory

        # Index by emotions
        for emotion in memory.emotions:
            if emotion not in self.emotional_index:
                self.emotional_index[emotion] = []
            self.emotional_index[emotion].append(memory_id)

        # Index by timestamp
        date_key = memory.timestamp.date()
        if date_key not in self.temporal_index:
            self.temporal_index[date_key] = []
        self.temporal_index[date_key].append(memory_id)

        # Establish connections with related memories
        self.connections[memory_id] = memory.related_memories

    def find_related_memories(self, memory_id: str, max_depth: int = 2) -> List[Memory]:
        """Find related memories up to a certain depth of connections"""
        related = set()
        current_depth = {memory_id}

        for _ in range(max_depth):
            next_depth = set()
            for current_id in current_depth:
                if current_id in self.connections:
                    next_depth.update(self.connections[current_id])
            related.update(next_depth)
            current_depth = next_depth

        return [self.memories[mid] for mid in related if mid in self.memories]


class DocumentManager:
    """Manages document creation and versioning"""

    def __init__(self, output_dir: str = "generated_documents"):
        self.output_dir = output_dir
        self.documents: Dict[str, Document] = {}
        os.makedirs(output_dir, exist_ok=True)

    def create_document(self, doc: Document) -> str:
        """Create a new document and return its ID"""
        doc_id = f"{doc.type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.documents[doc_id] = doc

        # Save document
        doc_path = os.path.join(self.output_dir, f"{doc_id}.json")
        with open(doc_path, 'w') as f:
            json.dump({
                'title': doc.title,
                'content': doc.content,
                'type': doc.type,
                'created_at': doc.created_at.isoformat(),
                'author': doc.author,
                'metadata': doc.metadata,
                'associated_memories': doc.associated_memories,
                'version_history': doc.version_history
            }, f, indent=2)

        return doc_id


class ContextGenerator:
    """Generates rich context for memories and documents using LLM"""

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key)

        self.location_contexts = {
            'work': ['office', 'meeting room', 'cafeteria', 'conference center'],
            'home': ['living room', 'kitchen', 'home office', 'garden'],
            'social': ['restaurant', 'park', 'cafe', 'friend\'s house'],
            'travel': ['airport', 'hotel', 'tourist spot', 'train station']
        }

        self.emotion_sets = {
            'positive': ['joy', 'excitement', 'satisfaction', 'pride'],
            'negative': ['frustration', 'anxiety', 'disappointment', 'worry'],
            'neutral': ['curiosity', 'focus', 'contemplation', 'calmness']
        }

    def generate_memory_context(self, base_context: str) -> Dict:
        """Generate rich context for a memory"""
        prompt = f"""Given the base context: "{base_context}"
        Generate rich sensory and emotional details for this memory. Include:
        1. Visual details
        2. Sounds
        3. Smells or tastes if relevant
        4. Physical sensations
        5. Emotional state
        6. Key participants
        Format as JSON with these keys: sensory_details, emotions, participants"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Generate rich memory context as JSON"},
                {"role": "user", "content": prompt}
            ]
        )

        try:
            return json.loads(response.choices[0].message.content)
        except:
            # Fallback to basic context if API fails
            return {
                "sensory_details": {"visual": base_context},
                "emotions": ["neutral"],
                "participants": ["self"]
            }


class EpisodicGenerator:
    """Main class for generating interconnected memories and documents"""

    def __init__(self, api_key: Optional[str] = None):
        self.memory_graph = MemoryGraph()
        self.document_manager = DocumentManager()
        self.context_generator = ContextGenerator(api_key)
        self.current_time = datetime.now()

    def generate_memory_sequence(self,
                                 start_date: datetime,
                                 num_days: int,
                                 memories_per_day: range) -> List[Memory]:
        """Generate a sequence of interconnected memories over time"""
        all_memories = []
        current_date = start_date

        for _ in range(num_days):
            # Generate 1-5 memories for this day
            num_memories = random.randint(memories_per_day.start, memories_per_day.stop)

            for _ in range(num_memories):
                # Generate base context
                context = self._generate_base_context(current_date)

                # Enrich context using LLM
                rich_context = self.context_generator.generate_memory_context(context)

                # Create memory
                memory = Memory(
                    timestamp=current_date + timedelta(hours=random.randint(8, 20)),
                    context=context,
                    location=self._select_location(),
                    participants=rich_context['participants'],
                    emotions=rich_context['emotions'],
                    sensory_details=rich_context['sensory_details'],
                    significance=random.randint(1, 10),
                    related_memories=[],  # Will be filled in later
                    tags=self._generate_tags(context)
                )

                # Add to graph
                memory_id = f"memory_{current_date.strftime('%Y%m%d')}_{len(all_memories)}"
                self.memory_graph.add_memory(memory_id, memory)
                all_memories.append(memory)

                # Generate associated document if significant
                if memory.significance > 7:
                    self._generate_associated_document(memory, memory_id)

            current_date += timedelta(days=1)

        # Establish connections between related memories
        self._establish_memory_connections(all_memories)

        return all_memories

    def _generate_base_context(self, date: datetime) -> str:
        """Generate base context for a memory"""
        contexts = [
            "Had an important meeting about {topic}",
            "Made significant progress on {project}",
            "Encountered an unexpected challenge with {issue}",
            "Celebrated {event} with colleagues",
            "Learned something new about {subject}"
        ]

        topics = [
            "project planning", "team dynamics", "technical architecture",
            "client requirements", "process improvement", "innovation initiatives"
        ]

        template = random.choice(contexts)
        topic = random.choice(topics)

        return template.format(
            topic=topic,
            project=f"the {topic} project",
            issue=f"the {topic} situation",
            event=f"success in {topic}",
            subject=topic
        )

    def _select_location(self) -> str:
        """Select a location for the memory"""
        context_type = random.choice(list(self.context_generator.location_contexts.keys()))
        return random.choice(self.context_generator.location_contexts[context_type])

    def _generate_tags(self, context: str) -> List[str]:
        """Generate relevant tags for the memory"""
        # Extract key terms from context
        words = context.lower().split()
        tags = [word for word in words if len(word) > 4]  # Simple but could be more sophisticated

        # Add some general tags
        general_tags = ["work", "meeting", "learning", "collaboration", "challenge"]
        tags.extend(random.sample(general_tags, 2))

        return list(set(tags))  # Remove duplicates

    def _establish_memory_connections(self, memories: List[Memory]):
        """Establish connections between related memories"""
        for i, memory in enumerate(memories):
            # Find memories with similar tags
            related = []
            for j, other in enumerate(memories):
                if i != j:
                    common_tags = set(memory.tags) & set(other.tags)
                    if len(common_tags) >= 2:  # At least 2 common tags
                        related.append(f"memory_{other.timestamp.strftime('%Y%m%d')}_{j}")

            # Update memory's related_memories list
            memory.related_memories = related[:3]  # Limit to 3 related memories

    def _generate_associated_document(self, memory: Memory, memory_id: str):
        """Generate a document associated with a significant memory"""
        doc_types = ["note", "email", "report", "diary"]
        doc_type = random.choice(doc_types)

        # Generate document content based on memory
        content = self._generate_document_content(memory, doc_type)

        document = Document(
            title=f"{doc_type.capitalize()}: {memory.context[:50]}...",
            content=content,
            type=doc_type,
            created_at=memory.timestamp,
            author="System",
            metadata={
                "significance": memory.significance,
                "location": memory.location,
                "emotions": memory.emotions
            },
            associated_memories=[memory_id],
            version_history=[{
                "version": 1,
                "timestamp": memory.timestamp.isoformat(),
                "changes": "Initial creation"
            }]
        )

        self.document_manager.create_document(document)

    def _generate_document_content(self, memory: Memory, doc_type: str) -> str:
        """Generate appropriate content for different document types"""
        if doc_type == "email":
            return f"""Subject: Re: {memory.context}

Dear Team,

I wanted to document the key points from our recent {memory.context.lower()}:

Location: {memory.location}
Key Participants: {', '.join(memory.participants)}

Key Points:
1. {memory.sensory_details.get('visual', 'Visual observation')}
2. {memory.sensory_details.get('sounds', 'Discussion points')}

Next Steps:
- Follow up on key decisions
- Schedule follow-up meeting
- Document action items

Best regards,
[System]"""

        elif doc_type == "diary":
            return f"""Dear Diary,

Today was significant ({memory.significance}/10). While at {memory.location}, 
{memory.context}. I felt {', '.join(memory.emotions)}.

What stood out:
{memory.sensory_details.get('visual', '')}
{memory.sensory_details.get('sounds', '')}

People involved: {', '.join(memory.participants)}

Reflections:
This experience made me think about its impact on future projects and team dynamics.

[End Entry]"""

        else:  # note or report
            return f"""## Summary Report

Event: {memory.context}
Date: {memory.timestamp.strftime('%Y-%m-%d %H:%M')}
Location: {memory.location}
Significance: {memory.significance}/10

### Key Details
{memory.sensory_details.get('visual', '')}

### Participants
{', '.join(memory.participants)}

### Observations
- Emotional context: {', '.join(memory.emotions)}
- Physical environment: {memory.sensory_details.get('sounds', '')}

### Next Steps
1. Document key learnings
2. Share with relevant team members
3. Schedule follow-up if needed

[End Report]"""


# Example usage
if __name__ == "__main__":
    generator = EpisodicGenerator()

    # Generate a week of memories
    start_date = datetime.now() - timedelta(days=7)
    memories = generator.generate_memory_sequence(
        start_date=start_date,
        num_days=365,
        memories_per_day=range(2, 9)  # 1-5 memories per day
    )

    # Print summary
    print(f"Generated {len(memories)} memories over 7 days")
    print("\nSample memory details:")
    for memory in random.sample(memories, 3):
        print(f"\nDate: {memory.timestamp}")
        print(f"Context: {memory.context}")
        print(f"Location: {memory.location}")
        print(f"Emotions: {', '.join(memory.emotions)}")
        print(f"Significance: {memory.significance}/10")
        print(f"Tags: {', '.join(memory.tags)}")