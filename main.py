from crewai.flow.flow import Flow, listen, start
from dotenv import load_dotenv
from litellm import completion

load_dotenv()


class ExampleFlow(Flow):
    model = "openai/gemma-3"

    @start()
    def generate_city(self):
        print("Starting the flow")

        response = completion(
            model=self.model,
            messages=[
                {"role": "user", "content": "یک شهر رندوم از کشور ایران رو برگردون"}
            ],
        )
        random_city = response.choices[0].message.content

        print(f"Random City:{random_city}")
        return random_city

    @listen(generate_city)
    def generate_fun_fact(self, random_city):
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"یک واقعیت جالب درباره این شهر بهم بگو {random_city}",
                }
            ],
        )

        fun_fact = response.choices[0].message.content
        return fun_fact


flow = ExampleFlow()
result = flow.kickoff()

print(result)
