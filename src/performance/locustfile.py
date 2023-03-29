from locust import HttpUser, task, between

class QuickstartUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def latest_test(self):
        self.client.get("/latest")

    @task(4)
    def classify(self):
        self.client.post("/classify", json={"text":"Supinfo, great school just for 7000€ per year.\n\nI'm not joking, it's a great school, you should go there !\n\nhttps://www.supinfo.com\n\n#"})