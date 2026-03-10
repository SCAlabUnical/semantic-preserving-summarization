
class StratifierRandom:

    def __init__(self, debug=False):

        self.debug = debug

    def stratify(self,df, DIM, M):

        sample_df = df.sample(n=M)

        return sample_df