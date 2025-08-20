from crewai.flow.flow import Flow, listen, start


class ExampleFlow(Flow):
    @start()
    def generate_city(self):
        return "Qom"

    @listen(generate_city)
    def generate_fun_fact(self, random_city):
        pass


flow = ExampleFlow()
result = flow.kickoff()

print(f"generated fun fact for {result}")
