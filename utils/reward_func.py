
def dummy_reward_fn(text):

        return 1.0 if "##Result##" in text and '##Reason##' in text else 0.0