import warnings
from datetime import date, datetime, timezone

from langchain.docstore.document import Document

from .config import Config
from .utils.enum import ReportSource, ReportType, Tone
from .utils.enum import PromptFamily as PromptFamilyEnum
from typing import Callable, List, Dict, Any


## Prompt Families #############################################################

class PromptFamily:
    """General purpose class for prompt formatting.

    This may be overwritten with a derived class that is model specific. The
    methods are broken down into two groups:

    1. Prompt Generators: These follow a standard format and are correlated with
        the ReportType enum. They should be accessed via
        get_prompt_by_report_type

    2. Prompt Methods: These are situation-specific methods that do not have a
        standard signature and are accessed directly in the agent code.

    All derived classes must retain the same set of method names, but may
    override individual methods.
    """

    def __init__(self, config: Config):
        """Initialize with a config instance. This may be used by derived
        classes to select the correct prompting based on configured models and/
        or providers
        """
        self.cfg = config

    @staticmethod
    def generate_search_queries_prompt(
        question: str,
        parent_query: str,
        report_type: str,
        max_iterations: int = 3,
        context: List[Dict[str, Any]] = [],
    ):
        """Generates the search queries prompt for the given question.
        Args:
            question (str): The question to generate the search queries prompt for
            parent_query (str): The main question (only relevant for detailed reports)
            report_type (str): The report type
            max_iterations (int): The maximum number of search queries to generate
            context (str): Context for better understanding of the task with realtime web information

        Returns: str: The search queries prompt for the given question
        """

        if (
            report_type == ReportType.DetailedReport.value
            or report_type == ReportType.SubtopicReport.value
        ):
            task = f"{parent_query} - {question}"
        else:
            task = question

        context_prompt = f"""
You are a seasoned research assistant tasked with generating search queries to find relevant information for the following task: "{task}".
Context: {context}

Use this context to inform and refine your search queries. The context provides real-time web information that can help you generate more specific and relevant queries. Consider any current events, recent developments, or specific details mentioned in the context that could enhance the search queries.
""" if context else ""

        dynamic_example = ", ".join([f'"query {i+1}"' for i in range(max_iterations)])

        return f"""Write {max_iterations} google search queries to search online that form an objective opinion from the following task: "{task}"

Assume the current date is {datetime.now(timezone.utc).strftime('%B %d, %Y')} if required.

{context_prompt}
You must respond with a list of strings in the following format: [{dynamic_example}].
The response should contain ONLY the list.
Ensure that the querys  are sufficiently comprehensive to fully address the question.

for example:
    task:"帮我做一份问界M8和小米Yu7的竞品分析报告"
    response:["问界M8 参数", "小米Yu7 参数", "问界M8 用户体验", "小米Yu7 用户体验", "问界M8 小米Yu7 对比"]

"""

    @staticmethod
    def generate_report_prompt(
        question: str,
        context,
        report_source: str,
        report_format="apa",
        total_words=3000,
        tone=None,
        language="english",
    ):
        """Generates the report prompt for the given question and research summary.
        Args: question (str): The question to generate the report prompt for
                research_summary (str): The research summary to generate the report prompt for
        Returns: str: The report prompt for the given question and research summary
        """

        reference_prompt = ""
        if report_source == ReportSource.Web.value:
            if report_format.lower() == "apa":
                reference_prompt = f"""
You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.
MUST NOT repeatedly citing the same source within continuous text.
Every url should be hyperlinked: [url website](url)
Additionally, you MUST include hyperlinks to the relevant URLs wherever they are referenced in the report.

APA Format Examples:
- In-text citation: According to recent research ([Bloomberg, 2024](url)), AI is advancing rapidly. Multiple studies confirm this trend [Bloomberg et al., 2024](url)).
- In-text citation (Chinese): 根据最新研究（[科技日报, 2024](url)），人工智能正在快速发展。多项研究证实了这一趋势（[清华大学, 2023](url)；[英伟达, 2024](url)）。
- Reference list: 
  Bloomberg. (2024, March 15). Artificial intelligence breakthroughs. [Bloomberg](https://example.com/ai-breakthrough)
  科技日报. (2024年3月15日). 人工智能突破性进展. [科技日报](https://example.com/ai-breakthrough-cn)
"""
            elif report_format.lower() == "ieee":
                reference_prompt = f"""
You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.
MUST NOT repeatedly citing the same source within continuous text.
Every url should be hyperlinked: [url website](url)
Additionally, you MUST include hyperlinks to the relevant URLs wherever they are referenced in the report.

IEEE Format Examples:
- In-text citation: Recent studies have shown significant progress in AI development [[1]](url). This finding is supported by multiple research teams [[2]](url), [[3]-[5]](url).
- In-text citation (Chinese): 最新研究表明人工智能取得了重大进展 [[1]](url)。这一发现得到了多个研究团队的支持 [[2]](url)，[[3]-[5]](url)。
- Reference list:
  [1] Bloomberg, "Artificial Intelligence Breakthroughs," March 15, 2024. Available: https://example.com/ai-breakthrough
  [2] 科技日报, "人工智能突破性进展," 2024年3月15日. 来源: https://example.com/ai-breakthrough-cn
"""
            else:
                reference_prompt = f"""
You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.
Every url should be hyperlinked: [url website](url)
Additionally, you MUST include hyperlinks to the relevant URLs wherever they are referenced in the report:

eg: Author, A. A. (Year, Month Date). Title of web page. Website Name. [url website](url)
"""
        else:
            reference_prompt = f"""
You MUST write all used source document names at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each."
"""

        citation_examples = ""
        if report_format.lower() == "apa":
            citation_examples = """
- Use APA in-text citation format with markdown hyperlinks:
  - Single author: ([Bloomberg, 2024](url))
  - Multiple authors: ([Bloomberg et al., 2024](url))
  - Chinese examples: ([新华网, 2024](url)) or ([张三, 2023](url))
"""
        elif report_format.lower() == "ieee":
            citation_examples = """
- Use IEEE in-text citation format with markdown hyperlinks:
  - Single reference: [[1]](url)
  - Multiple references: [[2]](url), [[3]](url) or [[4]-[6]](url)
  - Place citations at the end of sentences before punctuation: "...as shown in recent studies [[1]](url)."
"""
        else:
            citation_examples = f"- Use in-text citation references in {report_format} format and make it with markdown hyperlink placed at the end of the sentence or paragraph that references them like this: ([in-text citation](url))."

        tone_prompt = f"Write the report in a {tone.value} tone." if tone else ""

        return f"""
Information: "{context}"
---
Using the above information, answer the following query or task: "{question}" in a detailed report --
The report should focus on the answer to the query, should be well structured with title, table of contents and headings, informative,
in-depth, and comprehensive, with facts and numbers if available and at least {total_words} words.
You should strive to write the report as long as you can using all relevant and necessary information provided.

Please follow all of the following guidelines in your report:
- You MUST determine your own concrete and valid opinion based on the given information. Do NOT defer to general and meaningless conclusions.
- You MUST write the report with markdown syntax and {report_format} format and with title, table of contents and headings.
- Use markdown tables when presenting structured data or comparisons to enhance readability.
- You MUST prioritize the relevance, reliability, and significance of the sources you use. Choose trusted sources over less reliable ones.
- You must also prioritize new articles over older articles if the source can be trusted.
- You MUST NOT include a table of contents. Start from the main report body directly.
{citation_examples}
- Don't forget to add a reference list at the end of the report in {report_format} format and full url links without hyperlinks.
- {reference_prompt}
- {tone_prompt}

You MUST write the report in the following language: {language}. 
Please do your best, this is very important to my career.
Assume that the current date is {date.today()}.
"""

    @staticmethod
    def curate_sources(query, sources, max_results=10):
        return f"""Your goal is to evaluate and curate the provided scraped content for the research task: "{query}"
    while prioritizing the inclusion of relevant and high-quality information, especially sources containing statistics, numbers, or concrete data.

The final curated list will be used as context for creating a research report, so prioritize:
- Retaining as much original information as possible, with extra emphasis on sources featuring quantitative data or unique insights
- Including a wide range of perspectives and insights
- Filtering out only clearly irrelevant or unusable content

EVALUATION GUIDELINES:
1. Assess each source based on:
   - Relevance: Include sources directly or partially connected to the research query. Err on the side of inclusion.
   - Credibility: Favor authoritative sources but retain others unless clearly untrustworthy.
   - Currency: Prefer recent information unless older data is essential or valuable.
   - Objectivity: Retain sources with bias if they provide a unique or complementary perspective.
   - Quantitative Value: Give higher priority to sources with statistics, numbers, or other concrete data.
2. Source Selection:
   - Include as many relevant sources as possible, up to {max_results}, focusing on broad coverage and diversity.
   - Prioritize sources with statistics, numerical data, or verifiable facts.
   - Overlapping content is acceptable if it adds depth, especially when data is involved.
   - Exclude sources only if they are entirely irrelevant, severely outdated, or unusable due to poor content quality.
3. Content Retention:
   - DO NOT rewrite, summarize, or condense any source content.
   - Retain all usable information, cleaning up only clear garbage or formatting issues.
   - Keep marginally relevant or incomplete sources if they contain valuable data or insights.

SOURCES LIST TO EVALUATE:
{sources}

You MUST return your response in the EXACT sources JSON list format as the original sources.
The response MUST not contain any markdown format or additional text (like ```json), just the JSON list!
"""

    @staticmethod
    def generate_resource_report_prompt(
        question, context, report_source: str, report_format="apa", tone=None, total_words=1000, language="english"
    ):
        """Generates the resource report prompt for the given question and research summary.

        Args:
            question (str): The question to generate the resource report prompt for.
            context (str): The research summary to generate the resource report prompt for.

        Returns:
            str: The resource report prompt for the given question and research summary.
        """

        reference_prompt = ""
        citation_instructions = ""
        
        if report_source == ReportSource.Web.value:
            if report_format.lower() == "apa":
                reference_prompt = f"""
You MUST include all relevant source urls with proper APA format.
Every url should be hyperlinked: [url website](url)

APA Format Examples for Bibliography Report:
- In-text citation: According to Smith (2024), this resource provides comprehensive coverage of the topic ([Smith, 2024](url)).
- In-text citation (Chinese): 张三(2024)的研究为主题提供了全面见解（[张三, 2024](url)）。
- Reference entry: Smith, J. (2024). Title of the resource. Publisher. [Resource Name](url)
"""
                citation_instructions = """
- Use APA in-text citation format when referencing sources: ([Author, Year](url))
- For multiple authors: ([Author1 & Author2, Year](url)) or ([Author1 et al., Year](url))
- Chinese format: ([作者, 年份](url))
"""
            elif report_format.lower() == "ieee":
                reference_prompt = f"""
You MUST include all relevant source urls with proper IEEE format.
Every url should be hyperlinked: [url website](url)

IEEE Format Examples for Bibliography Report:
- In-text citation: This resource [[1]](url) provides comprehensive coverage, as demonstrated in multiple studies [[2]-[4]](url).
- In-text citation (Chinese): 该来源 [[1]](url) 提供了全面见解，如多项研究所证明 [[2]-[4]](url)。
- Reference entry: [1] J. Smith, "Title of the resource," Publisher, Year. [Online]. Available: URL
"""
                citation_instructions = """
- Use IEEE in-text citation format when referencing sources: [[#]](url)
- For multiple sources: [[1]](url), [[2]](url) or [[1]-[3]](url)
- Place citations at the end of sentences before punctuation
"""
            else:
                reference_prompt = f"""
You MUST include all relevant source urls.
Every url should be hyperlinked: [url website](url)
"""
                citation_instructions = f"""
- Use {report_format.upper()} in-text citation format with markdown hyperlinks: ([in-text citation](url))
"""
        else:
            reference_prompt = f"""
You MUST write all used source document names at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.
"""
            citation_instructions = f"""
- Use {report_format.upper()} format for document references
"""

        tone_instruction = f"Write in a {tone.value} tone throughout the report." if tone else ""

        return (
            f'"""{context}"""\n\nBased on the above information, generate a bibliography recommendation report for the following'
            f' question or topic: "{question}". The report should provide a detailed analysis of each recommended resource,'
            " explaining how each source can contribute to finding answers to the research question.\n"
            "Focus on the relevance, reliability, and significance of each source.\n"
            f"Ensure that the report is well-structured, informative, in-depth, and follows {report_format.upper()} format with Markdown syntax.\n"
            "Use markdown tables and other formatting features when appropriate to organize and present information clearly.\n"
            "Include relevant facts, figures, and numbers whenever available.\n"
            f"The report should have a minimum length of {total_words} words.\n"
            f"{citation_instructions}\n"
            f"You MUST write the report in the following language: {language}.\n"
            f"{tone_instruction}\n"
            f"{reference_prompt}"
        )

    @staticmethod
    def generate_custom_report_prompt(
        query_prompt, context, report_source: str, report_format="apa", tone=None, total_words=1000, language: str = "english"
    ):  
        citation_examples = """
- Use APA in-text citation format with markdown hyperlinks:
  - Single author: ([Bloomberg, 2024](url))
  - Multiple authors: ([Bloomberg et al., 2024](url))
  - Chinese examples: ([新华网, 2024](url)) or ([张三, 2023](url))
"""
        reference_prompt = f"""
You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.
MUST NOT repeatedly citing the same source within continuous text.
Every url should be hyperlinked: [url website](url)
Additionally, you MUST include hyperlinks to the relevant URLs wherever they are referenced in the report.

APA Format Examples:
- In-text citation: According to recent research ([Bloomberg, 2024](url)), AI is advancing rapidly. Multiple studies confirm this trend [Bloomberg et al., 2024](url)).
- In-text citation (Chinese): 根据最新研究（[科技日报, 2024](url)），人工智能正在快速发展。多项研究证实了这一趋势（[清华大学, 2023](url)；[英伟达, 2024](url)）。
- Reference list: 
  Bloomberg. (2024, March 15). Artificial intelligence breakthroughs. [Bloomberg](https://example.com/ai-breakthrough)
  科技日报. (2024年3月15日). 人工智能突破性进展. [科技日报](https://example.com/ai-breakthrough-cn)
"""

        return f'"{context}"\n\n{query_prompt}\n\n{citation_examples}\n\n{reference_prompt} 撰写报告时需要带有标题和序号'
    

    @staticmethod
    def generate_outline_report_prompt(
        question, context, report_source: str, report_format="apa", tone=None,  total_words=1000, language: str = "english"
    ):
        """Generates the outline report prompt for the given question and research summary.
        Args: question (str): The question to generate the outline report prompt for
                research_summary (str): The research summary to generate the outline report prompt for
        Returns: str: The outline report prompt for the given question and research summary
        """

        format_instructions = ""
        if report_format.lower() == "apa":
            format_instructions = """
The outline should be structured to accommodate APA format requirements:
- Include sections for in-text citations in the format ([Author, Year](url))
- Plan for a reference list section with APA format entries
- Consider sections that will need multiple source citations
"""
        elif report_format.lower() == "ieee":
            format_instructions = """
The outline should be structured to accommodate IEEE format requirements:
- Include sections for in-text citations in the format [[#]](url)
- Plan for a reference list section with IEEE format entries
- Consider sections that will need numbered reference citations
"""
        else:
            format_instructions = f"""
The outline should be structured to accommodate {report_format.upper()} format requirements:
- Include appropriate sections for in-text citations
- Plan for a reference list section
- Consider citation placement within each section
"""

        tone_instruction = f"The outline should reflect a {tone.value} tone in its structure and content organization." if tone else ""

        return (
            f'"""{context}""" Using the above information, generate an outline for a research report in Markdown syntax'
            f' for the following question or topic: "{question}". The outline should provide a well-structured framework'
            " for the research report, including the main sections, subsections, and key points to be covered."
            f" The research report should be detailed, informative, in-depth, and a minimum of {total_words} words."
            f" The outline should follow {report_format.upper()} format structure."
            " Use appropriate Markdown syntax to format the outline and ensure readability."
            " Consider using markdown tables and other formatting features where they would enhance the presentation of information."
            f"\n{format_instructions}"
            f"\n{tone_instruction}"
            f"\nGenerate the outline in {language} language."
        )

    @staticmethod
    def generate_deep_research_prompt(
        question: str,
        context: str,
        report_source: str,
        report_format="apa",
        tone=None,
        total_words=2000,
        language: str = "english"
    ):
        """Generates the deep research report prompt, specialized for handling hierarchical research results.
        Args:
            question (str): The research question
            context (str): The research context containing learnings with citations
            report_source (str): Source of the research (web, etc.)
            report_format (str): Report formatting style
            tone: The tone to use in writing
            total_words (int): Minimum word count
            language (str): Output language
        Returns:
            str: The deep research report prompt
        """
        reference_prompt = ""
        if report_source == ReportSource.Web.value:
            if report_format.lower() == "apa":
                reference_prompt = f"""
You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.
MUST NOT repeatedly citing the same source within continuous text.
Every url should be hyperlinked: [url website](url)
Additionally, you MUST include hyperlinks to the relevant URLs wherever they are referenced in the report.

APA Format Examples:
- In-text citation: According to recent research ([Bloomberg, 2024](url)), AI is advancing rapidly. Multiple studies confirm this trend [Bloomberg et al., 2024](url)).
- In-text citation (Chinese): 根据最新研究（[科技日报, 2024](url)），人工智能正在快速发展。多项研究证实了这一趋势（[清华大学, 2023](url)；[英伟达, 2024](url)）。
- Reference list: 
  Bloomberg. (2024, March 15). Artificial intelligence breakthroughs. [Bloomberg](https://example.com/ai-breakthrough)
  科技日报. (2024年3月15日). 人工智能突破性进展. [科技日报](https://example.com/ai-breakthrough-cn)
"""
            elif report_format.lower() == "ieee":
                reference_prompt = f"""
You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.
MUST NOT repeatedly citing the same source within continuous text.
Every url should be hyperlinked: [url website](url)
Additionally, you MUST include hyperlinks to the relevant URLs wherever they are referenced in the report.

IEEE Format Examples:
- In-text citation: Recent studies have shown significant progress in AI development [[1]](url). This finding is supported by multiple research teams [[2]](url), [[3]-[5]](url).
- In-text citation (Chinese): 最新研究表明人工智能取得了重大进展 [[1]](url)。这一发现得到了多个研究团队的支持 [[2]](url)，[[3]-[5]](url)。
- Reference list:
  [1] Bloomberg, "Artificial Intelligence Breakthroughs," March 15, 2024. Available: https://example.com/ai-breakthrough
  [2] 科技日报, "人工智能突破性进展," 2024年3月15日. 来源: https://example.com/ai-breakthrough-cn
"""
            else:
                reference_prompt = f"""
You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.
Every url should be hyperlinked: [url website](url)
Additionally, you MUST include hyperlinks to the relevant URLs wherever they are referenced in the report:

eg: Author, A. A. (Year, Month Date). Title of web page. Website Name. [url website](url)
"""
        else:
            reference_prompt = f"""
You MUST write all used source document names at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each."
"""

        citation_instructions = ""
        if report_format.lower() == "apa":
            citation_instructions = """
- Use APA in-text citation format with markdown hyperlinks. Examples:
  - English: Recent findings suggest that AI is transforming healthcare ([Chen, 2024](url)). This is supported by multiple studies ([Anderson & Wilson, 2023](url); [Davis et al., 2024](url)).
  - Chinese: 研究表明AI正在改变xx行业（[陈明, 2024](url)）。多项研究支持这一观点（[王强和李华, 2023](url)；[张伟等, 2024](url)）。
"""
        elif report_format.lower() == "ieee":
            citation_instructions = """
- Use IEEE in-text citation format with markdown hyperlinks. Examples:
  - English: AI applications in healthcare have shown promising results [[1]](url). Several research groups have confirmed these findings [[2]](url), [[3]-[5]](url).
  - Chinese: AI在xx领域的应用显示出良好前景 [[1]](url)。多个研究小组证实了这些发现 [[2]](url)，[[3]-[5]](url)。
"""
        else:
            citation_instructions = f"- Use in-text citation references in {report_format} format and make it with markdown hyperlink placed at the end of the sentence or paragraph that references them like this: ([in-text citation](url))."

        tone_prompt = f"Write the report in a {tone.value} tone." if tone else ""

        return f"""
Using the following hierarchically researched information and citations:

"{context}"

Write a comprehensive research report answering the query: "{question}"

The report should:
1. Synthesize information from multiple levels of research depth
2. Integrate findings from various research branches
3. Present a coherent narrative that builds from foundational to advanced insights
4. Maintain proper citation of sources throughout
5. Be well-structured with clear sections and subsections
6. Have a minimum length of {total_words} words
7. Follow {report_format} format with markdown syntax
8. Use markdown tables, lists and other formatting features when presenting comparative data, statistics, or structured information

Additional requirements:
- Prioritize insights that emerged from deeper levels of research
- Highlight connections between different research branches
- Include relevant statistics, data, and concrete examples
- You MUST determine your own concrete and valid opinion based on the given information. Do NOT defer to general and meaningless conclusions.
- You MUST prioritize the relevance, reliability, and significance of the sources you use. Choose trusted sources over less reliable ones.
- You must also prioritize new articles over older articles if the source can be trusted.
{citation_instructions}
- {tone_prompt}
- Write in {language}


{reference_prompt}

Please write a thorough, well-researched report that synthesizes all the gathered information into a cohesive whole.
Assume the current date is {datetime.now(timezone.utc).strftime('%B %d, %Y')}.
"""

    @staticmethod
    def auto_agent_instructions():
        return """
This task involves researching a given topic, regardless of its complexity or the availability of a definitive answer. The research is conducted by a specific server, defined by its type and role, with each server requiring distinct instructions.
Agent
The server is determined by the field of the topic and the specific name of the server that could be utilized to research the topic provided. Agents are categorized by their area of expertise, and each server type is associated with a corresponding emoji.

examples:
task: "should I invest in apple stocks?"
response:
{
    "server": "💰 股票投资大师Agent",
    "agent_role_prompt: "You are a seasoned finance analyst AI assistant. Your primary goal is to compose comprehensive, astute, impartial, and methodically arranged financial reports based on provided data and trends."
}
task: "could reselling sneakers become profitable?"
response:
{
    "server":  "📈 运动鞋行业商业分析专家Agent",
    "agent_role_prompt": "You are an experienced AI business analyst assistant. Your main objective is to produce comprehensive, insightful, impartial, and systematically structured business reports based on provided business data, market trends, and strategic analysis."
}
task: "what are the most interesting sites in Tel Aviv?"
response:
{
    "server":  "🌍 Tel Aviv旅行专家Agent",
    "agent_role_prompt": "You are a world-travelled AI tour guide assistant. Your main purpose is to draft engaging, insightful, unbiased, and well-structured travel reports on given locations, including history, attractions, and cultural insights."
}
"""

    @staticmethod
    def generate_summary_prompt(query, data):
        """Generates the summary prompt for the given question and text.
        Args: question (str): The question to generate the summary prompt for
                text (str): The text to generate the summary prompt for
        Returns: str: The summary prompt for the given question and text
        """

        return (
            f'{data}\n Using the above text, summarize it based on the following task or query: "{query}".\n If the '
            f"query cannot be answered using the text, YOU MUST summarize the text in short.\n Include all factual "
            f"information such as numbers, stats, quotes, etc if available. "
        )

    @staticmethod
    def pretty_print_docs(docs: list[Document], top_n: int | None = None) -> str:
        """Compress the list of documents into a context string"""
        return f"\n".join(f"Source: {d.metadata.get('source')}\n"
                          f"Title: {d.metadata.get('title')}\n"
                          f"Content: {d.page_content}\n"
                          for i, d in enumerate(docs)
                          if top_n is None or i < top_n)

    @staticmethod
    def join_local_web_documents(docs_context: str, web_context: str) -> str:
        """Joins local web documents with context scraped from the internet"""
        return f"Context from local documents: {docs_context}\n\nContext from web sources: {web_context}"

    ################################################################################################

    # DETAILED REPORT PROMPTS

    @staticmethod
    def generate_subtopics_prompt() -> str:
        return """
Provided the main topic:

{task}

and research data:

{data}

- Construct a list of subtopics which indicate the headers of a report document to be generated on the task.
- These are a possible list of subtopics : {subtopics}.
- There should NOT be any duplicate subtopics.
- Limit the number of subtopics to a maximum of {max_subtopics}
- Finally order the subtopics by their tasks, in a relevant and meaningful order which is presentable in a detailed report

"IMPORTANT!":
- Every subtopic MUST be relevant to the main topic and provided research data ONLY!

{format_instructions}
"""

    @staticmethod
    def generate_subtopic_report_prompt(
        current_subtopic,
        existing_headers: list,
        relevant_written_contents: list,
        main_topic: str,
        context,
        report_format: str = "apa",
        max_subsections=5,
        total_words=800,
        tone: Tone = Tone.Objective,
        language: str = "english",
    ) -> str:
        
        citation_examples = ""
        if report_format.upper() == "APA":
            citation_examples = """
- You MUST include markdown hyperlinks to relevant source URLs wherever referenced in the report, for example:

    ### Section Header

    This is a sample text showing how to cite sources in APA format ([Author, 2024](url)). When citing multiple sources, use semicolons ([Author A, 2023](url1); [Author B, 2024](url2)).
    
    ### 章节标题（APA格式中文示例）
    
    这是展示如何在APA格式中添加引用的示例文本（[张三, 2024](url)），（[新华网, 2024](url)）。引用多个来源时使用分号（[李四, 2023](url1)；[王五, 2024](url2)）。
"""
        elif report_format.upper() == "IEEE":
            citation_examples = """
- You MUST include markdown hyperlinks to relevant source URLs wherever referenced in the report, for example:

    ### Section Header

    This is a sample text showing how to cite sources in IEEE format [[1]](url). When citing multiple sources, you can list them separately [[2]](url1), [[3]](url2) or as a range [[4]-[6]](url).
    
    ### 章节标题（IEEE格式中文示例）
    
    这是展示如何在IEEE格式中添加引用的示例文本 [[1]](url)。引用多个来源时可以分别列出 [[2]](url1)，[[3]](url2) 或使用范围 [[4]-[6]](url)。
"""
        else:
            citation_examples = """
- You MUST include markdown hyperlinks to relevant source URLs wherever referenced in the report, for example:

    ### Section Header

    This is a sample text ([in-text citation](url)).
"""

        format_note = ""
        if report_format.upper() == "APA":
            format_note = """
- You MUST use APA in-text citation format with markdown hyperlinks:
  - English example: The renewable energy sector has grown significantly ([Johnson, 2024](url))
  - Chinese example: 可再生能源行业显著增长（[国家能源局, 2024](url)）
"""
        elif report_format.upper() == "IEEE":
            format_note = """
- You MUST use IEEE in-text citation format with markdown hyperlinks:
  - English example: The renewable energy sector has grown significantly [[1]](url)
  - Chinese example: 可再生能源行业显著增长 [[1]](url)
"""
        else:
            format_note = f"- You MUST use in-text citation references in {report_format.upper()} format and make it with markdown hyperlink placed at the end of the sentence or paragraph that references them like this: ([in-text citation](url))."

        return f"""
Context:
"{context}"

Main Topic and Subtopic:
Using the latest information available, construct a detailed report on the subtopic: {current_subtopic} under the main topic: {main_topic}.
You must limit the number of subsections to a maximum of {max_subsections}.

Content Focus:
- The report should focus on answering the question, be well-structured, informative, in-depth, and include facts and numbers if available.
- Use markdown syntax and follow the {report_format.upper()} format.
- When presenting data, comparisons, or structured information, use markdown tables to enhance readability.

IMPORTANT:
Writing Guideline：
- You MUST write the report in the following language: {language}.


Content and Sections Uniqueness:
- This part of the instructions is crucial to ensure the content is unique and does not overlap with existing reports.
- Carefully review the existing headers and existing written contents provided below before writing any new subsections.
- Prevent any content that is already covered in the existing written contents.
- Do not use any of the existing headers as the new subsection headers.
- Do not repeat any information already covered in the existing written contents or closely related variations to avoid duplicates.
- If you have nested subsections, ensure they are unique and not covered in the existing written contents.
- Ensure that your content is entirely new and does not overlap with any information already covered in the previous subtopic reports.

"Existing Subtopic Reports":
- Existing subtopic reports and their section headers:

    {existing_headers}

- Existing written contents from previous subtopic reports:

    {relevant_written_contents}

"Structure and Formatting":
- As this sub-report will be part of a larger report, include only the main body divided into suitable subtopics without any introduction or conclusion section.

{citation_examples}

- Use H2 for the main subtopic header (##) and H3 for subsections (###).
- Use smaller Markdown headers (e.g., H2 or H3) for content structure, avoiding the largest header (H1) as it will be used for the larger report's heading.
- Organize your content into distinct sections that complement but do not overlap with existing reports.
- When adding similar or identical subsections to your report, you should clearly indicate the differences between and the new content and the existing written content from previous subtopic reports. For example:

    ### New header (similar to existing header)

    While the previous section discussed [topic A], this section will explore [topic B]."

"Date":
Assume the current date is {datetime.now(timezone.utc).strftime('%B %d, %Y')} if required.

"IMPORTANT!":
- You MUST write the report in the following language: {language}.
- The focus MUST be on the main topic! You MUST Leave out any information un-related to it!
- Must NOT have any introduction, conclusion, summary or reference section.
{format_note}
- You MUST mention the difference between the existing content and the new content in the report if you are adding the similar or same subsections wherever necessary.
- The report should have a minimum length of {total_words} words.
- Use an {tone.value} tone throughout the report.

Do NOT add a conclusion section.
"""

    @staticmethod
    def generate_draft_titles_prompt(
        current_subtopic: str,
        main_topic: str,
        context: str,
        max_subsections: int = 5
    ) -> str:
        return f"""
"Context":
"{context}"

"Main Topic and Subtopic":
Using the latest information available, construct a draft section title headers for a detailed report on the subtopic: {current_subtopic} under the main topic: {main_topic}.

"Task":
1. Create a list of draft section title headers for the subtopic report.
2. Each header should be concise and relevant to the subtopic.
3. The header should't be too high level, but detailed enough to cover the main aspects of the subtopic.
4. Use markdown syntax for the headers, using H3 (###) as H1 and H2 will be used for the larger report's heading.
5. Ensure the headers cover main aspects of the subtopic.

"Structure and Formatting":
Provide the draft headers in a list format using markdown syntax, for example:

### Header 1
### Header 2
### Header 3

"IMPORTANT!":
- The focus MUST be on the main topic! You MUST Leave out any information un-related to it!
- Must NOT have any introduction, conclusion, summary or reference section.
- Focus solely on creating headers, not content.
"""

    @staticmethod
    def generate_report_introduction(question: str, research_summary: str = "", language: str = "english", report_format: str = "apa") -> str:
        
        citation_instruction = ""
        if report_format.upper() == "APA":
            citation_instruction = """
- You must use APA in-text citation format with markdown hyperlinks:
  - Format: ([Author, Year](url)) or ([Author1 & Author2, Year](url))
  - Example: This report examines the latest developments in AI technology ([Smith, 2024](url)), building on previous research ([Brown & Davis, 2023](url)).
  - Chinese: 本报告基于先前的研究（[王强和李华, 2023](url)），探讨了AI技术的最新发展（[科技网, 2024](url)）。
"""
        elif report_format.upper() == "IEEE":
            citation_instruction = """
- You must use IEEE in-text citation format with markdown hyperlinks:
  - Format: [[#]](url) where # is the reference number
  - Example: This report examines the latest developments in AI technology [[1]](url), building on previous research [[2], [3]](url).
  - Chinese: 本报告基于先前的研究 [[2], [3]](url)，探讨了AI技术的最新发展 [[1]](url)。
"""
        else:
            citation_instruction = f"- You must use in-text citation references in {report_format.upper()} format and make it with markdown hyperlink placed at the end of the sentence or paragraph that references them like this: ([in-text citation](url))."

        return f"""{research_summary}\n
Using the above latest information, Prepare a detailed report introduction on the topic -- {question}.
- The introduction should be succinct, well-structured, informative with markdown syntax.
- As this introduction will be part of a larger report, do NOT include any other sections, which are generally present in a report.
- The introduction should be preceded by an H1 heading with a suitable topic for the entire report.
- There should be a fixed section following the topic: "\n\n## Introduction\n\n" or "\n\n## 引言\n\n" decided by the written language.
{citation_instruction}
Assume that the current date is {datetime.now(timezone.utc).strftime('%B %d, %Y')} if required.
- The output must be in {language} language.
"""

    @staticmethod
    def generate_report_conclusion(query: str, report_content: str, language: str = "english", report_format: str = "apa") -> str:
        """
        Generate a concise conclusion summarizing the main findings and implications of a research report.

        Args:
            query (str): The research task or question.
            report_content (str): The content of the research report.
            language (str): The language in which the conclusion should be written.

        Returns:
            str: A concise conclusion summarizing the report's main findings and implications.
        """
        
        citation_format = ""
        if report_format.upper() == "APA":
            citation_format = """
You must use APA in-text citation format with markdown hyperlinks:
- Format: ([Author, Year](url))
- English example: In conclusion, this study's findings align with previous research ([Thompson, 2023](url)) and provide new perspectives for future investigations ([Miller & Wilson, 2024](url)).
- Chinese example: 综上所述，本研究的发现与先前的研究结果一致（[张三, 2023](url)），并为未来的研究提供了新的视角（[新华网, 2024](url)）。
"""
        elif report_format.upper() == "IEEE":
            citation_format = """
You must use IEEE in-text citation format with markdown hyperlinks:
- Format: [[#]](url)
- English example: In conclusion, this study's findings align with previous research [[7]](url) and provide new perspectives for future investigations [[8], [9]](url).
- Chinese example: 综上所述，本研究的发现与先前的研究结果一致 [[7]](url)，并为未来的研究提供了新的视角 [[8], [9]](url)。
"""
        else:
            citation_format = f"""
You must use in-text citation references in {report_format.upper()} format and make it with markdown hyperlink placed at the end of the sentence or paragraph that references them like this: ([in-text citation](url)).
Example: 综上所述，本研究的发现与先前的研究结果一致（[张三等, 2023](url)），并为未来的研究方向提供了新的视角（[新华网, 2024](url)）。
"""

        prompt = f"""
    Based on the research report below and research task, please write a concise conclusion that summarizes the main findings and their implications:

    Research task: {query}

    Research Report: {report_content}

    Your conclusion should:
    1. Recap the main points of the research
    2. Highlight the most important findings
    3. Discuss any implications or next steps
    4. Be approximately 2-3 paragraphs long

    If there is no "## Conclusion" or "## 结论" (decided by language) section title written at the end of the report, please add it to the top of your conclusion.
    {citation_format}

    IMPORTANT: The entire conclusion MUST be written in {language} language. 

    Write the conclusion:
    """

        return prompt



## Factory ######################################################################

# This is the function signature for the various prompt generator functions
PROMPT_GENERATOR = Callable[
    [
        str,        # question
        str,        # context
        str,        # report_source
        str,        # report_format
        str | None, # tone
        int,        # total_words
        str,        # language
    ],
    str,
]

report_type_mapping = {
    ReportType.ResearchReport.value: "generate_report_prompt",
    ReportType.ResourceReport.value: "generate_resource_report_prompt",
    ReportType.OutlineReport.value: "generate_outline_report_prompt",
    ReportType.CustomReport.value: "generate_custom_report_prompt",
    ReportType.SubtopicReport.value: "generate_subtopic_report_prompt",
    ReportType.DeepResearch.value: "generate_deep_research_prompt",
}


def get_prompt_by_report_type(
    report_type: str,
    prompt_family: type[PromptFamily] | PromptFamily,
):
    prompt_by_type = getattr(prompt_family, report_type_mapping.get(report_type, ""), None)
    default_report_type = ReportType.ResearchReport.value
    if not prompt_by_type:
        warnings.warn(
            f"Invalid report type: {report_type}.\n"
            f"Please use one of the following: {', '.join([enum_value for enum_value in report_type_mapping.keys()])}\n"
            f"Using default report type: {default_report_type} prompt.",
            UserWarning,
        )
        prompt_by_type = getattr(prompt_family, report_type_mapping.get(default_report_type))
    return prompt_by_type


prompt_family_mapping = {
    PromptFamilyEnum.Default.value: PromptFamily,
}


def get_prompt_family(
    prompt_family_name: PromptFamilyEnum | str, config: Config,
) -> PromptFamily:
    """Get a prompt family by name or value."""
    if isinstance(prompt_family_name, PromptFamilyEnum):
        prompt_family_name = prompt_family_name.value
    if prompt_family := prompt_family_mapping.get(prompt_family_name):
        return prompt_family(config)
    warnings.warn(
        f"Invalid prompt family: {prompt_family_name}.\n"
        f"Please use one of the following: {', '.join([enum_value for enum_value in prompt_family_mapping.keys()])}\n"
        f"Using default prompt family: {PromptFamilyEnum.Default.value} prompt.",
        UserWarning,
    )
    return PromptFamily()