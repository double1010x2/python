import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

class genQubitMat():

    def __init__(self, n_qubit) -> None:
        self.n_qubit = n_qubit
        self.n_qstate = 2**n_qubit
        self.n_dim   = int(np.sqrt(self.n_qstate))
        self.n_dim_qb= int(np.log2(self.n_dim))
        self.qmat    = None
        pass

    def genQMat(self):

        sub_qmat = np.zeros((self.n_dim, self.n_dim), np.int)
        qmat     = np.zeros((self.n_dim_qb*6, self.n_dim, self.n_dim), np.int)

        qi = 0
        qb = 0
        n_split = self.n_dim
        while 2**qb < self.n_dim:
            n_split = int(self.n_dim / 2**(qb+1))
            n_loop   = int(self.n_dim / n_split)
            # 1st type  
            istart = 0
            for ni in range(n_loop):
                qmat[qi][istart:istart+n_split, :] = ni % 2
                istart += n_split
            qmat[qi+1] = -qmat[qi]+1

            # 2nd type  
            istart = 0
            for ni in range(n_loop):
                qmat[qi+2][:, istart:istart+n_split] = ni % 2
                istart += n_split
            qmat[qi+3] = -qmat[qi+2]+1

            # 3rd type
            qmat[qi+4] = np.abs(qmat[qi]-qmat[qi+2])
            qmat[qi+5] = -qmat[qi+4]+1 

            qi += 6
            qb += 1
        self.qmat = qmat

if __name__ == "__main__":
    cqmat = genQubitMat(10)
    cqmat.genQMat()
    ig, axes = plt.subplots(ncols=6, nrows=5)
    qi = 0
    for i, ax in enumerate(axes.flatten()):
        ax.matshow(cqmat.qmat[qi])
#        ax.set_title('qMat {}'.format(qi))
        qi += 1
    plt.tight_layout()
    plt.show()
    import pdb; pdb.set_trace()
"""
    Aone = np.ones((2,2), np.int)

    q1   = np.identity(2, np.int)
    #q2   = np.rot90(q1)
    q3   = np.zeros_like(q1)
    q3[:,0] = 1
    #q4   = np.fliplr(q3)
    q5   = np.zeros_like(q1)
    q5[0,:] = 1
    #q6   = np.flipud(q5)
    qmat = [q1, q3, q5]

    ndim = qmat[0].shape[0]
    import pdb; pdb.set_trace()
    while ndim < 32:
        Aone = np.ones((ndim, ndim), np.int)
        qmat_new = []
        for qq in qmat:
            qmat_new.append(np.kron(Aone, qq))
            qmat_new.append(np.kron(qq, Aone))
        qmat = qmat_new
        ndim = qq.shape[0]

    import pdb; pdb.set_trace()
    q1   = np.kron(q1, Aone)
    q11  = np.kron(Aone, q1)
    q3   = np.kron(q3, Aone)
    q33  = np.kron(Aone, q3)
    q5   = np.kron(q5, Aone)
    q55  = np.kron(Aone, q5)
    import pdb; pdb.set_trace()
    """