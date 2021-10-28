from AC2test import *

class TestACDistributedMinibatch():
    def init_args(self):
        self.args = {'layer_sizes': [2, 21, 21, 21, 21, 1],
                    'run_functions_eagerly': False,
                    'epoch_adam': 20,
                    'epoch_lbfgs': 20,
                    'dist_training': True,
                    'dict_adaptive': {"residual": [False],
                                      "BCs": [True, False]},
                    'N_x': 100,
                    'N_t': 50,
                    'N_f': 5000,
                    'batch_sz': 200,
                    }

    def test_lgfgs_eager1(self):
        self.init_args()
        self.args['lbfgs_eager'] = True
        self.args['isAdaptive']= False
        try:
            main(self.args)
            assert True
        except Exception as inst:
            if "Currently we dont support distributed minibatching training" in inst.args[0]:
                assert True
            else:
                assert False


    def test_lgfgs_eager2(self):
        self.init_args()
        self.args['lbfgs_eager'] = False
        self.args['isAdaptive'] = False
        try:
            main(self.args)
            assert True
        except Exception as inst:
            if "Currently we dont support distributed minibatching training" in inst.args[0]:
                assert True
            else:
                assert False

    def test_adaptive1(self):
        self.init_args()
        self.args['lbfgs_eager'] = True
        self.args['isAdaptive'] = True
        self.args['dict_adaptive']= {"residual": [True],
                                     "BCs": [False, False]}
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
                                     "BCs": [True, True]}
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
                                     "BCs": [True, False]}
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
                                     "BCs": [False, True]}
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
                                     "BCs": [False, False]}
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
                                     "BCs": [True, True]}
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
                                     "BCs": [True, False]}
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
                                     "BCs": [False, True]}
        try:
            main(self.args)
            assert True
        except Exception as inst:
            if "Currently we dont support distributed training for adaptive PINNs" in inst.args[0]:
                assert True
            else:
                assert False
