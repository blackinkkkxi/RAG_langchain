import re
import json
from openai import OpenAI

class Agent:
    def __init__(self, name, agent_prompt):
        self.name = name
        self.agent_prompt = agent_prompt
        self.client = OpenAI()
        self.messages = []
        self.steps = []

    def get_solution(self, problem, previous_solution=None):
        self.initialize_conversation(problem, previous_solution)
        prompt = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in self.messages])
        response = self.generate_response(prompt)
        
        if not response:
            print(f"{self.name} did not return a valid response.")

        step_data = self.parse_response(response)

        if not step_data:
            print(f"{self.name} failed to parse response: {response}")

        self.store_step(step_data)

        return self.compile_solution()

    def initialize_conversation(self, problem, previous_solution):
        self.messages = []
        # Initialize system prompt
        self.messages.append({"role": "system", "content": self.agent_prompt})
        
        # Add problem and previous solution (if any)
        user_message = f"问题:\n{problem}\n"
        if previous_solution:
            user_message += f"\n之前的解决方案:\n{previous_solution}\n请继续进行你的推理分析。"
        else:
            user_message += "\n请提供你的解决方案。"

        self.messages.append({"role": "user", "content": user_message})
        # Assistant initial acknowledgment
        self.messages.append({"role": "assistant", "content": "好的，我现在开始推理"})

    def generate_response(self, prompt):
        # Generate response from the model
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None

    def parse_response(self, result):
        # Attempt to parse the assistant's response as JSON
        try:
            json_pattern = r"```json\s*(\{[\s\S]*?\})\s*```"
            match = re.findall(json_pattern, result)
            if match:
                json_list = []
                for temp_match in match:
                    json_list.append(json.loads(temp_match))
                return json_list
        except json.JSONDecodeError as e:
            print(f"{self.name} error parsing JSON response: {str(e)} - Response Text: {result}")
        return None

    def store_step(self, step_data):
        # Store the parsed step data and append it to the conversation
        for data in step_data:
            self.steps.append((data['title'], data["content"]))
            self.messages.append({"role": "assistant", "content": data["content"]})

    def compile_solution(self):
        # Compile the steps into a final solution format
        return "\n\n".join([f"### {title}\n{content}" for title, content in self.steps])
