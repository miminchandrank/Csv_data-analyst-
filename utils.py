from typing import Dict, Any, List
import re


class AnswerFormatter:
    """Smart response formatter with context-aware rules"""

    def __init__(self):
        self.simple_questions = [
            r'columns?( names| list)?',
            r'how many (rows|columns)',
            r'data types?',
            r'shape( of|$)',
            r'(is|are) there (duplicates|missing)',
            r'what (is|are) the (first|last) \d+ rows'
        ]

        self.stats_questions = [
            r'(mean|median|average|max|min) of',
            r'unique values in',
            r'most frequent'
        ]

    def normalize_question(self, question: str) -> str:
        """Standardize question format"""
        if not isinstance(question, str):
            return ""
        question = question.lower().strip('? ')
        question = re.sub(r'\bwhat\'?s\b', 'what is', question)
        return re.sub(r'\b(please|can you|show me)\b', '', question)

    def is_simple_question(self, question: str) -> bool:
        """Check if question requires only direct answer"""
        norm_q = self.normalize_question(question)
        return any(re.search(pattern, norm_q) for pattern in self.simple_questions)

    def is_stats_question(self, question: str) -> bool:
        """Check if question is about basic statistics"""
        norm_q = self.normalize_question(question)
        return any(re.search(pattern, norm_q) for pattern in self.stats_questions)

    def format_response(self, answer: Dict[str, Any], question: str) -> str:
        """Context-aware response formatting"""
        if not answer or not isinstance(answer, dict):
            return "Error: Invalid answer format"

        norm_q = self.normalize_question(question)

        # 1. Simple factual questions
        if self.is_simple_question(norm_q):
            return answer.get('answer', 'No answer found')

        # 2. Statistical questions (answer + brief context)
        elif self.is_stats_question(norm_q):
            response = answer.get('answer', '')
            if answer.get('source_documents'):
                first_doc = answer['source_documents'][0]
                content = first_doc if isinstance(first_doc, str) else first_doc.page_content
                response += f"\n\n(Source: {content[:150]}...)"
            return response

        # 3. Analytical questions (full response)
        else:
            response = f"Analysis: {answer.get('answer', '')}\n\n"
            if answer.get('source_documents'):
                response += "Supporting Evidence:\n"
                for i, doc in enumerate(answer['source_documents'][:2], 1):
                    content = doc if isinstance(doc, str) else doc.page_content
                    response += f"{i}. {content[:200]}...\n"
            return response


def generate_data_summary(summary: Dict[str, Any]) -> str:
    """Enhanced data summary generator with error handling"""
    if not summary or not isinstance(summary, dict):
        return "Error: Invalid summary format"

    try:
        report = []
        shape = summary.get('metadata', {}).get('shape', (0, 0))
        columns = summary.get('metadata', {}).get('columns', [])

        report.append(f"📊 Dataset Summary ({shape[0]} rows × {shape[1]} columns)")
        report.append(f"\n🔡 Columns ({len(columns)} total):")
        report.append(", ".join(columns[:5]) + ("..." if len(columns) > 5 else ""))

        report.append("\n📝 Data Types:")
        dtypes = summary.get('metadata', {}).get('dtypes', {})
        type_counts = {}
        for dtype in dtypes.values():
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        for dtype, count in type_counts.items():
            report.append(f"- {dtype}: {count} columns")

        return "\n".join(report)
    except Exception as e:
        return f"Error generating summary: {str(e)}"