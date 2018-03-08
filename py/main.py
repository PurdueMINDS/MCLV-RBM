from __future__ import print_function

import itertools
from multiprocessing.pool import Pool

from r.rbm import *
from r.rbmutil import RBMUtil
from u.config import *
from u.gpu_provider import GpuProvider
from u.phase import Phase
from u.sqlite import SQLite

np.set_printoptions(threshold=0, suppress=True)


class Main:
    @classmethod
    def main(cls):
        pool = Pool(processes=GpuProvider.gpu_count)
        phase = Phase[ARGS.phase]
        if phase == Phase.RUN_TOURS:
            gpu = GpuProvider.get_gpu('run_tours')
            for fn in ARGS.filename:
                rbm = RBMUtil.load_model(fn, gpu)
                rbm.generate_name()
                rbm.gpu = gpu
                data, test = cls.load_data(gpu)
                for sample_uniform in [True, False]:
                    RBMUtil.run_tours(rbm, data, sample_uniform)
            return
        elif phase == Phase.GRADIENT:
            hidden_to_try = [(num_hidden, phase)
                             for _, num_hidden in itertools.product(range(int(ARGS.runs))
                                                                    , [int(i) for i in ARGS.num_hidden])
                             ]
            pool.map(cls.run_gradient_or_partition, hidden_to_try)

        elif phase == Phase.DATA:
            models_to_try = [
                cls.get_rbm(method=method
                            , num_hidden=num_hidden
                            , learning_rate=learning_rate
                            , schedule=schedule
                            , k=k
                            , batch_size=batch_size)
                for iteration, num_hidden, learning_rate, schedule, k, batch_size, method in itertools.product(
                    range(int(ARGS.runs))
                    , [int(i) for i in ARGS.num_hidden]
                    , [float(i) for i in ARGS.learning_rate]
                    , [Schedule[i] for i in ARGS.schedule]
                    , [int(i) for i in ARGS.k]
                    , [int(i) for i in ARGS.batch_size]
                    , [Method[i] for i in ARGS.method]
                )]
            pool.map(cls.run_data, models_to_try)

    @classmethod
    def run_data(cls, rbm):
        gpu = None
        try:
            gpu = GpuProvider.get_gpu(rbm.qualified_name)
            rbm.gpu = gpu

            data, test = cls.load_data(gpu)
            rbm = rbm.fit(data, test)
            if LOG_TOUR:
                RBMUtil.run_tours(rbm, data)
            filename, _ = RBMUtil.save_model(rbm, rbm.name)
            log_Z, L, Lt = RBMUtil.compute_likelihood(rbm, data, test)
            SQLite.insert_dict(FINAL_LIKELIHOODS_TABLE, Util.dictize(
                name=rbm.name
                , model_location=filename
                , log_Z=log_Z
                , L_train=L
                , L_test=Lt
                , iteration=0
                , iterations=len(range(int(ARGS.runs)))
            ))
        except Exception as e:
            Log.exception(e)
            if LOCAL:
                raise e
        finally:
            if gpu is not None:
                GpuProvider.return_gpu(gpu, rbm.qualified_name)

    @classmethod
    def run_gradient_or_partition(cls, num_hidden, phase):
        gpu = None
        try:
            gpu = GpuProvider.get_gpu(num_hidden)
            data, test = cls.load_data(gpu)
            rbm = cls.get_poorly_trained_rbm(data, num_hidden, test, gpu)
            for method, k, batch_size in itertools.product(
                    [Method[i] for i in ARGS.method]
                    , [int(i) for i in ARGS.k]
                    , [int(i) for i in ARGS.batch_size]
            ):
                rbm.analyze_gradients(data, method, k, batch_size)
        except Exception as e:
            Log.exception(e)
            if LOCAL:
                raise e
        finally:
            if gpu is not None:
                GpuProvider.return_gpu(gpu, num_hidden)

    @classmethod
    def load_data(cls, gpu):
        data = np.load(MNIST_FOLDER + "data.npy")
        test = np.load(MNIST_FOLDER + "test.npy")
        data = data[:10000] if LOCAL else data
        data = Util.add_bias_coefficient(data).astype(NP_DT)
        test = Util.add_bias_coefficient(test).astype(NP_DT)
        data = gpu.from_numpy(data)
        test = gpu.from_numpy(test)
        Log.var(data_length=len(data))
        if DISCRETE:
            data = Util.mle_discretize(data)
            test = Util.mle_discretize(test)
        return data, test

    @classmethod
    def get_poorly_trained_rbm(cls, data, num_hidden, test, gpu):
        fitted = cls.get_rbm(method=Method.CD
                             , num_hidden=num_hidden
                             , learning_rate=0.1
                             , schedule=Schedule.EXP100
                             , k=1
                             , batch_size=128
                             , gpu=gpu).fit(
            data, test)
        RBMUtil.save_model(fitted, fitted.name)
        return fitted

    @classmethod
    def get_rbm(cls, method, num_hidden, learning_rate, schedule, k, batch_size, gpu=None):
        cdk = k
        if Method.requires_warmup(method):
            mclvk = k
            cdk_warmup = int(ARGS.cdk_warmup)
            if cdk_warmup > 0:
                cdk = cdk_warmup
        else:
            mclvk = -1
        return RBM(
            WIDTH * HEIGHT,
            num_hidden,
            warmup_epochs=int(ARGS.warmup_epochs),
            cdk=cdk,
            mclvk=mclvk,
            weight_decay=float(ARGS.weight_decay),
            momentum=float(ARGS.momentum),
            learning_rate=learning_rate,
            schedule=schedule,
            max_epochs=int(ARGS.total_epochs),
            batch_size=batch_size,
            method=method,
            supernode_samples=int(ARGS.supernode_samples),
            mat_nd=int(ARGS.mat_nd[0]),
            mat_batch_size=ARGS.mat_batch_size,
            mat_ssf=ARGS.mat_ssf,
            gpu=gpu
        )


if __name__ == "__main__":
    print(ARGS)
    Main().main()
