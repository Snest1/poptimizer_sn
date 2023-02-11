"""Формирование примеров для обучения в формате PyTorch."""
from typing import Any, Dict, List, Tuple, Type, Union

import pandas as pd
from torch import Tensor
from torch.utils import data

from poptimizer.dl import features

import logging
#import dumper


LOGGER = logging.getLogger()


from pymongo.collection import Collection
from poptimizer.store.database import DB, MONGO_CLIENT
_COLLECTION_speedy = MONGO_CLIENT[DB]["sn_speedy"]
def snspeedy_get_collection() -> Collection:
    return _COLLECTION_speedy

# Описание фенотипа и его подразделов
PhenotypeData = Dict[str, Union[Any, "PhenotypeData"]]

#sn_global_params: features.DataParams
sn_global_params = []

def f(ticker):
#restore
    import pickle
    import bson
    collection_speedy = snspeedy_get_collection()
#    record = collection_speedy.find_one({'_id': ''})
    record = collection_speedy.find_one({'name': 'params'})
#    restored = pickle.loads(record["bin_var1"])
    sn_speedy_params = pickle.loads(record["bin_var1"])
#    collection_speedy.delete_one({'_id': sn_speedy_params})
#   params = restored
    print("!!!!!!!!!!! ALL IS GOOD")


    return OneTickerDataset(ticker, sn_speedy_params)


class OneTickerDataset(data.Dataset):
    """Готовит обучающие примеры для одного тикера на основе параметров модели."""

    def __init__(self, ticker: str, params: features.DataParams):
        self.len = params.len(ticker)
        self.features = [
            getattr(features, feat_name)(ticker, params) for feat_name in params.get_all_feat()
        ]

    def __getitem__(self, item) -> Dict[str, Union[Tensor, List[Tensor]]]:
        example = {}
        for feature in self.features:
            key = feature.__class__.__name__
            example[key] = feature[item]
        return example

    def __len__(self) -> int:
        return self.len

    @property
    def features_description(self) -> Dict[str, Tuple[features.FeatureType, int]]:
        """Словарь с описанием всех признаков."""
        features_description = {}
        for feature in self.features:
            key = feature.__class__.__name__
            features_description[key] = feature.type_and_size
        return features_description


class DescribedDataLoader(data.DataLoader):
    """Загрузчик данных, который дополнительно хранит описание параметров данных."""

    def __init__(
        self,
        tickers: Tuple[str, ...],
        end: pd.Timestamp,
        params: PhenotypeData,
        params_type: Type[features.DataParams],
        sn_speedy_params: str = None,

    ):
        """Формирует загрузчики данных для обучения, валидации, тестирования и прогнозирования для
        заданных тикеров и конечной даты на основе словаря с параметрами.

        :param tickers:
            Перечень тикеров, для которых будет строится модель.
        :param end:
            Конец диапазона дат статистики, которые будут использоваться для
            построения модели.
        :param params:
            Словарь с параметрами для построения признаков и других элементов модели.
        :param params_type:
            Тип формируемых признаков.
        """
#        LOGGER.info(f"SNEDIT_024: Beg DescribedDataLoader")

#        print(f"SNEDIT_201: sn_speedy_params={sn_speedy_params}")


#        params = PhenotypeData()

        if 1==2 and sn_speedy_params:
#restore
            import pickle
            import bson
            collection_speedy = snspeedy_get_collection()
            record = collection_speedy.find_one({'_id': sn_speedy_params})
            restored = pickle.loads(record["bin_var1"])
            collection_speedy.delete_one({'_id': sn_speedy_params})
#            params = restored
            LOGGER.info("!!!!!!!!!!! ALL IS GOOD")


        params = params_type(tickers, end, params)


#save
#        import pickle
#        import bson
#        collection_speedy = snspeedy_get_collection()
#        speedy_id = collection_speedy.insert_one({
#            "name": "params",
#            "bin_var1": bson.Binary(pickle.dumps(    params    )),
##            "bin_var2": bson.Binary(pickle.dumps(state_dict)),
#        })



#        print(type(params), params)
#        quit()
#        sn_global_params = params
#        LOGGER.info(f"SNEDIT_024_1: Befor OneTickerDataSet modidfied")
#        sn_data_sets = [OneTickerDataset(ticker, params) for ticker in tickers]

        from multiprocessing import Pool


#        print(__name__)
        if 1==2 or __name__ == '__main__':
            p = Pool(20)
            print(p.map(f, tickers))

#        quit()
#        LOGGER.info(f"SNEDIT_024_2: Befor OneTickerDataSet standart")

        data_sets = [OneTickerDataset(ticker, params) for ticker in tickers]


#        LOGGER.info(f"SNEDIT_025: Befor Num_workers")

#        dumper.max_depth = 50
#        dumper.instance_dump = 'all'
#        dumper.dump(data_sets)

        super().__init__(
            dataset=data.ConcatDataset(data_sets),
            batch_size=params.batch_size,
            shuffle=params.shuffle,
            drop_last=False,
            num_workers=0,  # 
#            num_workers=24,  # Загрузка в отдельных потокых. Но увеличение потоков не добавляет производительности, а увеличивает кол-во процессов PYTHON3
        )
#        LOGGER.info(f"SNEDIT_026: After Num_workers")

        self._features_description = data_sets[0].features_description
        self._history_days = params.history_days

    @property
    def features_description(self) -> Dict[str, Tuple[features.FeatureType, int]]:
        """Словарь с описанием всех признаков."""
        return self._features_description

    @property
    def history_days(self) -> int:
        """Количество дней в истории."""
        return self._history_days
