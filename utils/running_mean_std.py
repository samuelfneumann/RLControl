import numpy as np
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count        
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count    

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var)

    def denormalize(self, x):
        return x * np.sqrt(self.var) + self.mean


def test_runningmeanstd():
    for (x1, x2, x3) in [
        (np.array([[-0.5,5]]), np.array([[0.0, 5]]), np.array([[0.5, 5]]))
        ]:

        rms = RunningMeanStd(epsilon=1e-4, shape=x1.shape[1:])

        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [x.mean(axis=0), x.var(axis=0)]
        rms.update(x1)
        rms.update(x2)
        rms.update(x3)
        ms2 = [rms.mean, rms.var]

        print( np.allclose(ms1, ms2))
        print('x1',x1)
        print('x2', x2)
        print('x3', x3)
        print('ms1', ms1)
        print('ms2',ms2)
        print(rms.normalize(x1[0]))
        print(rms.normalize(x2[0]))
        print(rms.normalize(x3[0]))

        print(rms.denormalize(rms.normalize(x1[0])))
        print(rms.denormalize(rms.normalize(x2[0])))
        print(rms.denormalize(rms.normalize(x3[0])))