import json
import requests
import rospy

class BayesianRepulsionUpdater:
    OPENAI_URL = "https://api.openai.com/v1/chat/completions"
    MODEL_NAME = "gpt-3.5-turbo"

    def __init__(self, api_key, obstacles):
        self.api_key = api_key
        self.obstacles = obstacles  # List of objects with .name, .repulsive_gain, .prob_coef
        self._priors = {}

        if not self.api_key.strip():
            raise ValueError("API key is missing.")

        self._initialize_priors()

    def _initialize_priors(self):
        for obs in self.obstacles:
            prior = max(0.0, min(1.0, obs.prob_coef))
            self._priors[obs] = prior

    def perform_update(self):
        
        names = sorted(set(obs.name for obs in self._priors.keys()))
        
        prompt_lines = [
            "You are evaluating obstacle danger implictily given the use prompts and the obstacle name from it. So if the enviornment is busy or dynamic or there are ongoing activities, the probability should increase and if theyre slow or empty or inactive it should decrease.",
            "Work station is very very busy today, go to destination",
            "For each of these object names, return a danger likelihood (0.0–1.0) in pure JSON:"
        ]
        
        for n in names:
            prompt_lines.append(f"- {n}")
        
        prompt_lines.append("") 
        prompt_lines.append("Example JSON format:")
        prompt_lines.append('{ "Crane": 0.8, "Barrier": 0.3 }')
        
        # Join with newlines
        prompt = "\n".join(prompt_lines)

        # Prepare request payload
        request_data = {
            "model": self.MODEL_NAME,
            "messages": [
                {"role": "system", "content": prompt}
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Send request to OpenAI
        response = requests.post(
            self.OPENAI_URL,
            headers=headers,
            data=json.dumps(request_data)
        )

        if not response.ok:
            rospy.loginfo(f"[OpenAI ERROR] {response.status_code} - {response.text}")
            return

        try:
            raw_content = response.json()["choices"][0]["message"]["content"]
            likelihood_map = json.loads(raw_content)
        except Exception as e:
            rospy.loginfo("[PARSE ERROR] Failed to parse GPT response:")
            rospy.loginfo(raw_content if 'raw_content' in locals() else response.text)
            return

        # Apply Bayes' rule 
        for obs, prior in self._priors.items():
            name = obs.name
            if name not in likelihood_map:
                rospy.loginfo(f"[WARN] No likelihood for '{name}', skipping.")
                continue

            likelihood = float(likelihood_map[name])
            
            # Bayes calculation
            num = prior * likelihood
            den = num + (1.0 - prior) * (1.0 - likelihood)
            post = num / den if den > 0.0 else prior

            # gain update
            obs.repulsive_gain = post * obs.repulsive_gain
            
            rospy.loginfo(f"[Bayes] {name}: prior={prior:.2f}, like={likelihood:.2f} and post={post:.2f}")

        rospy.loginfo("Bayesian update complete.")