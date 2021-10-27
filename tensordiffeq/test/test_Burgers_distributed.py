from Burgers2test import *

class TestBurgersDistributed():
    def init_args(self):
        self.args = {'layer_sizes': [2, 21, 21, 21, 21, 1],
                    'run_functions_eagerly': True,
                    'epoch_adam': 20,
                    'epoch_lbfgs': 20,
                    'lbfgs_eager': False,
                    'isAdaptive': True,
                    'dist_training': True,
                    'dict_adaptive': {"residual": [False],
                                      "BCs": [True, False, False]},
                    'N_x': 100,
                    'N_t': 50,
                    'N_f': 5000,
                    'batch_sz': None,
                    }

    def test_lbfgs_eager1(self):
        self.init_args()
        self.args['lbfgs_eager'] = True
        self.args['isAdaptive']= False
        try:
            main(self.args)
            assert True
        except Exception as inst:
            assert False

    def test_lbfgs_eager2(self):
        self.init_args()
        self.args['lbfgs_eager'] = False
        self.args['isAdaptive'] = False
        try:
            main(self.args)
            assert True
        except Exception as inst:
            assert False

    def test_adaptive1(self):
        self.init_args()
        self.args['lbfgs_eager'] = True
        self.args['isAdaptive'] = True
        self.args['dict_adaptive']= {"residual": [True],
                                     "BCs": [False, False, False]}
        try:
            main(self.args)
            assert True
        except Exception as inst:
            if "Currently we dont support distributed training for adaptive PINNs" in inst.args[0]:
                assert True
            else:
                assert False

    def test_adaptive2(self):
        self.init_args()
        self.args['lbfgs_eager'] = True
        self.args['isAdaptive'] = True
        self.args['dict_adaptive']= {"residual": [True],
                                     "BCs": [True, True, True]}
        try:
            main(self.args)
            assert True
        except Exception as inst:
            if "Currently we dont support distributed training for adaptive PINNs" in inst.args[0]:
                assert True
            else:
                assert False

    def test_adaptive3(self):
        self.init_args()
        self.args['lbfgs_eager'] = True
        self.args['isAdaptive'] = True
        self.args['dict_adaptive']= {"residual": [True],
                                     "BCs": [True, False, False]}
        try:
            main(self.args)
            assert True
        except Exception as inst:
            if "Currently we dont support distributed training for adaptive PINNs" in inst.args[0]:
                assert True
            else:
                assert False

    def test_adaptive4(self):
        self.init_args()
        self.args['lbfgs_eager'] = True
        self.args['isAdaptive'] = True
        self.args['dict_adaptive']= {"residual": [True],
                                     "BCs": [False, True, True]}
        try:
            main(self.args)
            assert True
        except Exception as inst:
            if "Currently we dont support distributed training for adaptive PINNs" in inst.args[0]:
                assert True
            else:
                assert False

    def test_adaptive5(self):
        self.init_args()
        self.args['lbfgs_eager'] = True
        self.args['isAdaptive'] = True
        self.args['dict_adaptive']= {"residual": [False],
                                     "BCs": [False, False, False]}
        try:
            main(self.args)
            assert True
        except Exception as inst:
            if "Currently we dont support distributed training for adaptive PINNs" in inst.args[0]:
                assert True
            else:
                assert False

    def test_adaptive6(self):
        self.init_args()
        self.args['lbfgs_eager'] = True
        self.args['isAdaptive'] = True
        self.args['dict_adaptive']= {"residual": [False],
                                     "BCs": [True, True, True]}
        try:
            main(self.args)
            assert True
        except Exception as inst:
            if "Currently we dont support distributed training for adaptive PINNs" in inst.args[0]:
                assert True
            else:
                assert False

    def test_adaptive7(self):
        self.init_args()
        self.args['lbfgs_eager'] = True
        self.args['isAdaptive'] = True
        self.args['dict_adaptive']= {"residual": [False],
                                     "BCs": [True, False, False]}
        try:
            main(self.args)
            assert True
        except Exception as inst:
            if "Currently we dont support distributed training for adaptive PINNs" in inst.args[0]:
                assert True
            else:
                assert False

    def test_adaptive8(self):
        self.init_args()
        self.args['lbfgs_eager'] = True
        self.args['isAdaptive'] = True
        self.args['dict_adaptive']= {"residual": [False],
                                     "BCs": [False, True, True]}
        try:
            main(self.args)
            assert True
        except Exception as inst:
            if "Currently we dont support distributed training for adaptive PINNs" in inst.args[0]:
                assert True
            else:
                assert False
