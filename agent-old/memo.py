class NL2SQLStateMachine:
    def __init__(self, llm, tools, config):
        self.llm = llm
        self.tools = tools
        self.config = config
        self.state = State.INIT
        self.memory = None
        
    def run(self, query: str) -> Dict[str, Any]:
        """메인 실행 루프"""
        self.memory = AgentMemory(query=query)
        self.state = State.PLANNING
        
        while self.state not in [State.SUCCESS, State.FAILED]:
            print(f"Current State: {self.state.value}")
            
            if self.state == State.PLANNING:
                self._planning_state()
            elif self.state == State.GATHERING:
                self._gathering_state()
            elif self.state == State.GENERATING:
                self._generating_state()
            elif self.state == State.VALIDATING:
                self._validating_state()
            elif self.state == State.EXECUTING:
                self._executing_state()
            elif self.state == State.ANALYZING_ERROR:
                self._analyzing_error_state()
            elif self.state == State.REFINING:
                self._refining_state()
                
        return self._get_result()
    
    def _planning_state(self):
        """LLM이 현재 상황을 파악하고 다음 행동을 결정"""
        prompt = self._build_planning_prompt()
        response = self.llm.generate(prompt)
        plan = self._parse_plan(response)
        
        self.memory.plan = plan
        
        # LLM의 결정에 따라 다음 상태 전환
        if plan['needs_schema'] or plan['needs_examples']:
            self.state = State.GATHERING
        else:
            self.state = State.GENERATING
    
    def _build_planning_prompt(self) -> str:
        """계획 수립을 위한 프롬프트"""
        return f"""
You are a SQL generation planning agent. Analyze the current situation and decide what to do next.

Natural Language Query: {self.memory.query}

Current Memory Status:
- Schema available: {self.memory.schema is not None}
- Similar examples available: {len(self.memory.similar_examples) > 0}
- Previous SQL generated: {self.memory.generated_sql is not None}
- Execution errors: {len(self.memory.error_history)}

Based on the query complexity and available information, create a plan.

Respond in JSON format:
{{
    "needs_schema": true/false,
    "needs_examples": true/false,
    "strategy": "direct_generation" | "example_based" | "iterative_refinement",
    "reasoning": "explanation of your decision",
    "confidence": 0.0-1.0
}}
"""

    def _gathering_state(self):
        """필요한 정보만 선택적으로 수집"""
        plan = self.memory.plan
        
        if plan['needs_schema'] and self.memory.schema is None:
            self.memory.schema = self.tools.get_schema()
            print("✓ Schema gathered")
        
        if plan['needs_examples'] and len(self.memory.similar_examples) == 0:
            self.memory.similar_examples = self.tools.search_similar_examples(
                self.memory.query,
                top_k=self.config.get('top_k_examples', 3)
            )
            print(f"✓ {len(self.memory.similar_examples)} similar examples gathered")
        
        self.state = State.GENERATING
    
    def _generating_state(self):
        """SQL 생성"""
        prompt = self._build_generation_prompt()
        sql = self.llm.generate(prompt)
        self.memory.generated_sql = self._extract_sql(sql)
        
        print(f"Generated SQL: {self.memory.generated_sql}")
        self.state = State.VALIDATING
    
    def _build_generation_prompt(self) -> str:
        """SQL 생성 프롬프트 - 컨텍스트 기반"""
        prompt_parts = [
            "Generate a SQL query for the following natural language question:",
            f"\nQuestion: {self.memory.query}\n"
        ]
        
        # Schema 정보
        if self.memory.schema:
            prompt_parts.append("\nDatabase Schema:")
            prompt_parts.append(self._format_schema(self.memory.schema))
        
        # Few-shot examples
        if self.memory.similar_examples:
            prompt_parts.append("\nSimilar Examples:")
            for i, example in enumerate(self.memory.similar_examples, 1):
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(f"Question: {example['question']}")
                prompt_parts.append(f"SQL: {example['sql']}")
        
        # 이전 에러 정보 (refinement 시)
        if self.memory.error_history:
            prompt_parts.append("\nPrevious Attempts and Errors:")
            for error in self.memory.error_history[-2:]:  # 최근 2개만
                prompt_parts.append(f"- SQL: {error['sql']}")
                prompt_parts.append(f"  Error: {error['error']}")
        
        prompt_parts.append("\nGenerate only the SQL query without explanation.")
        
        return "\n".join(prompt_parts)
    
    def _validating_state(self):
        """SQL 문법 검증"""
        sql = self.memory.generated_sql
        
        # 기본 문법 체크
        validation_result = self._basic_sql_validation(sql)
        
        if validation_result['valid']:
            print("✓ SQL validation passed")
            self.state = State.EXECUTING
        else:
            print(f"✗ SQL validation failed: {validation_result['error']}")
            self.memory.error_history.append({
                'sql': sql,
                'error': validation_result['error'],
                'type': 'validation'
            })
            self.state = State.ANALYZING_ERROR
    
    def _executing_state(self):
        """SQL 실행"""
        try:
            result = self.tools.execute_sql(self.memory.generated_sql)
            self.memory.execution_result = result
            print(f"✓ SQL executed successfully: {len(result)} rows")
            self.state = State.SUCCESS
            
        except Exception as e:
            print(f"✗ SQL execution failed: {str(e)}")
            self.memory.error_history.append({
                'sql': self.memory.generated_sql,
                'error': str(e),
                'type': 'execution'
            })
            self.state = State.ANALYZING_ERROR
    
    def _analyzing_error_state(self):
        """LLM을 활용한 에러 분석 및 수정 전략 수립"""
        if self.memory.refinement_attempts >= self.memory.max_refinement_attempts:
            print("Max refinement attempts reached")
            self.state = State.FAILED
            return
        
        # LLM에게 에러 분석 요청
        analysis_prompt = self._build_error_analysis_prompt()
        analysis = self.llm.generate(analysis_prompt)
        error_analysis = self._parse_error_analysis(analysis)
        
        print(f"Error Analysis: {error_analysis['diagnosis']}")
        print(f"Fix Strategy: {error_analysis['strategy']}")
        
        # 분석 결과에 따라 필요한 정보 추가 수집
        if error_analysis.get('needs_more_examples'):
            # 에러 타입에 맞는 예제 추가 검색
            additional_examples = self.tools.search_error_fix_examples(
                error_type=self.memory.error_history[-1]['type'],
                query=self.memory.query
            )
            self.memory.similar_examples.extend(additional_examples)
        
        self.memory.refinement_attempts += 1
        self.state = State.REFINING
    
    def _build_error_analysis_prompt(self) -> str:
        """에러 분석 프롬프트"""
        last_error = self.memory.error_history[-1]
        
        return f"""
You are a SQL debugging expert. Analyze the error and provide a fix strategy.

Original Question: {self.memory.query}

Database Schema:
{self._format_schema(self.memory.schema)}

Generated SQL:
{last_error['sql']}

Error Type: {last_error['type']}
Error Message: {last_error['error']}

Previous Attempts: {self.memory.refinement_attempts}

Analyze the error and respond in JSON format:
{{
    "diagnosis": "what caused the error",
    "error_category": "column_name" | "table_name" | "join_logic" | "aggregation" | "syntax" | "other",
    "strategy": "how to fix it",
    "needs_more_examples": true/false,
    "confidence": 0.0-1.0
}}
"""

    def _refining_state(self):
        """에러 분석을 바탕으로 SQL 재생성"""
        # 에러 컨텍스트를 포함한 프롬프트로 재생성
        prompt = self._build_refinement_prompt()
        refined_sql = self.llm.generate(prompt)
        self.memory.generated_sql = self._extract_sql(refined_sql)
        
        print(f"Refined SQL (attempt {self.memory.refinement_attempts}): {self.memory.generated_sql}")
        self.state = State.VALIDATING
    
    def _build_refinement_prompt(self) -> str:
        """정제를 위한 프롬프트 - 에러 컨텍스트 포함"""
        last_error = self.memory.error_history[-1]
        
        prompt_parts = [
            "Fix the SQL query based on the error analysis.",
            f"\nOriginal Question: {self.memory.query}",
            f"\nDatabase Schema:",
            self._format_schema(self.memory.schema),
            f"\nPrevious SQL (INCORRECT):",
            last_error['sql'],
            f"\nError: {last_error['error']}",
        ]
        
        # 수정 전략 힌트
        if self.memory.plan:
            prompt_parts.append(f"\nSuggested Fix Strategy:")
            prompt_parts.append(self.memory.plan.get('strategy', ''))
        
        # 유사한 에러 수정 예제
        if self.memory.similar_examples:
            prompt_parts.append("\nReference Examples:")
            for example in self.memory.similar_examples[:2]:
                prompt_parts.append(f"Question: {example['question']}")
                prompt_parts.append(f"SQL: {example['sql']}")
        
        prompt_parts.append("\nGenerate the corrected SQL query only.")
        
        return "\n".join(prompt_parts)
    
    # Helper methods
    def _basic_sql_validation(self, sql: str) -> Dict[str, Any]:
        """기본 SQL 문법 검증"""
        sql_upper = sql.upper().strip()
        
        if not sql_upper.startswith('SELECT'):
            return {'valid': False, 'error': 'SQL must start with SELECT'}
        
        if 'FROM' not in sql_upper:
            return {'valid': False, 'error': 'SQL must contain FROM clause'}
        
        # 괄호 매칭 체크
        if sql.count('(') != sql.count(')'):
            return {'valid': False, 'error': 'Unmatched parentheses'}
        
        return {'valid': True}
    
    def _format_schema(self, schema: Dict) -> str:
        """스키마를 읽기 쉬운 형식으로 포맷"""
        formatted = []
        for table, columns in schema.items():
            formatted.append(f"Table: {table}")
            formatted.append(f"Columns: {', '.join(columns)}")
        return "\n".join(formatted)
    
    def _extract_sql(self, response: str) -> str:
        """LLM 응답에서 SQL 추출"""
        # 코드 블록에서 추출
        if '```sql' in response:
            sql = response.split('```sql')[1].split('```')[0].strip()
        elif '```' in response:
            sql = response.split('```')[1].split('```')[0].strip()
        else:
            sql = response.strip()
        
        return sql
    
    def _parse_plan(self, response: str) -> Dict:
        """LLM의 계획 응답 파싱"""
        import json
        try:
            # JSON 추출 시도
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0].strip()
            elif '{' in response:
                json_str = response[response.find('{'):response.rfind('}')+1]
            else:
                json_str = response
            
            return json.loads(json_str)
        except:
            # 파싱 실패 시 기본 계획
            return {
                'needs_schema': True,
                'needs_examples': True,
                'strategy': 'example_based',
                'reasoning': 'Default plan due to parsing error',
                'confidence': 0.5
            }
    
    def _parse_error_analysis(self, response: str) -> Dict:
        """에러 분석 응답 파싱"""
        import json
        try:
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0].strip()
            elif '{' in response:
                json_str = response[response.find('{'):response.rfind('}')+1]
            else:
                json_str = response
            
            return json.loads(json_str)
        except:
            return {
                'diagnosis': 'Unable to parse error',
                'error_category': 'other',
                'strategy': 'Retry with more context',
                'needs_more_examples': True,
                'confidence': 0.3
            }
    
    def _get_result(self) -> Dict[str, Any]:
        """최종 결과 반환"""
        return {
            'state': self.state.value,
            'query': self.memory.query,
            'sql': self.memory.generated_sql,
            'result': self.memory.execution_result,
            'attempts': self.memory.refinement_attempts,
            'error_history': self.memory.error_history
        }