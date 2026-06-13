"""
Self-contained NBP stacker predictor that can be saved and used later.
Wraps:
- base model A (LogReg C=5 on word_12 + char_35 + extended_kw)
- base model F (LinearSVC C=2 on same features)
- base model E (CatBoost native text)
- rule features (8 groups derived from log-odds + FN analysis)
- meta LogReg (C=0.5) trained on OOF features + rules
- threshold τ
"""
import re, numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

NBP_KEYWORDS_EXTENDED = ['аукцион офз','банк россия','бюджет','бюджетный','бюджетный политика','бюджетный правило','валюта минфин','валюта центробанк','валюта фнб','внешний долг','госдолг','государство','дефицит','дефицит бюджет','доход','доходность офз','долг','долг рф','евробонд','ежедневный объём','золотовалютный','звр','историчский максимум','иностранный инвестор','край','ключевой','купить минфин','купонный','купонный доход','купон','кудрин','максимов','международный резерв','международный резервы','министерство финансы','министр','минфин','минфин валюта','минфин рамка','мишустин','монетарный','налог','налог вклад','налог доход','налоговый','ндс','ндфл','ндфл расчёт','нерезидент','нефтегазовый','нефтегазовый доход','объем','объём покупка','облигация федеральный','общий','орешкин','офз','офз пд','первичный размещение','платить','покупка валюта','правительство','продажа валюта','проект','развитие','район','расход','региональный','регион','резерв рф','рамка бюджетный','размещение','размещение офз','семья','сектор','силуанов','советник','счет','средство','средство фнб','сумма','уровень','урок','федеральный','федеральный заём','финансирование','финансы','фнб','фнс','цб минфин','доля нерезидент']
DKP_DISCRIMINATORS = ['ключевой ставка','решение ставка','ставка цб','процентный ставка','снижение ставка','повышение ставка','набиуллина','набиуллин','ужесточение дкп','смягчение дкп']

NBP_GROUP_FISCAL = [r'\bбюджетный\s+правило\b', r'\bдефицит\s+бюджет\b', r'\bфедеральный\s+бюджет\b',
                    r'\bбюджет.*политик', r'\bрасход.*бюджет\b', r'\bдоход.*бюджет\b']
NBP_GROUP_TAX = [r'\bналог.*вклад', r'\bндфл\b', r'\bналог.*доход\b', r'\bндс\b', r'\bфнс\b',
                 r'\bналоговый\s+служб', r'\bналог.*физических']
NBP_GROUP_DEBT = [r'\bофз\s+пд\b', r'\bаукцион\s+офз\b', r'\bразмещение\s+офз\b', r'\bдоля\s+нерезидент\b',
                  r'\bгосдолг\b', r'\bвнешн.*долг\b', r'\bевробонд\b', r'\bдолг\s+рф\b',
                  r'\bдолг\s+ро[сс]сия\b', r'\bобъём\s+размещение\b']
NBP_GROUP_RESERVES = [r'\bмеждународн.*резерв\b', r'\bзвр\b', r'\bзолотовалютн\b',
                      r'\bвалюта\s+фнб\b', r'\bсредство\s+фнб\b', r'\bмонетарн.*золот\b', r'\bзапас\s+золот\b']
NBP_GROUP_OPS = [r'\bвалюта\s+минфин\b', r'\bминфин\s+валюта\b', r'\bкупить\s+минфин\b',
                 r'\bкупон.*доход\b', r'\bкупонный\b', r'\bнефтегазовый\s+доход\b',
                 r'\bобязательн.*продаж.*выручк\b']
NBP_GROUP_PEOPLE = [r'\bсилуанов\b', r'\bмишустин\b', r'\bкудрин\b', r'\bорешкин\b', r'\bмаксимов\b',
                    r'\bминистр\s+финанс\b', r'\bминистерств.*финанс\b']
DKP_GROUP_RATE = [r'\bключевой\s+ставка\b', r'\bпроцентный\s+ставка\b', r'\bснижение\s+ставка\b',
                  r'\bповышение\s+ставка\b', r'\bбазис.*пункт\b']
DKP_GROUP_OTHER = [r'\bнабиуллина\b', r'\bнабиуллин\b', r'\bтремасов\b', r'\bдкп\b',
                   r'\bужесточен.*дкп\b', r'\bсмягчен.*дкп\b']

def rule_features_extended(X):
    out = np.zeros((len(X), 8))
    for i, t in enumerate(X):
        text = t.lower()
        out[i, 0] = sum(1 for r in NBP_GROUP_FISCAL if re.search(r, text))
        out[i, 1] = sum(1 for r in NBP_GROUP_TAX if re.search(r, text))
        out[i, 2] = sum(1 for r in NBP_GROUP_DEBT if re.search(r, text))
        out[i, 3] = sum(1 for r in NBP_GROUP_RESERVES if re.search(r, text))
        out[i, 4] = sum(1 for r in NBP_GROUP_OPS if re.search(r, text))
        out[i, 5] = sum(1 for r in NBP_GROUP_PEOPLE if re.search(r, text))
        out[i, 6] = sum(1 for r in DKP_GROUP_RATE if re.search(r, text))
        out[i, 7] = sum(1 for r in DKP_GROUP_OTHER if re.search(r, text))
    return out


class KeywordCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, kw_pos, kw_neg):
        self.kw_pos = kw_pos
        self.kw_neg = kw_neg
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        out = np.zeros((len(X), 2), dtype=float)
        for i, text in enumerate(X):
            text_lower = text.lower()
            n_tokens = max(1, len(text_lower.split()))
            pos_count = sum(text_lower.count(kw) for kw in self.kw_pos)
            neg_count = sum(text_lower.count(kw) for kw in self.kw_neg)
            out[i, 0] = pos_count / n_tokens
            out[i, 1] = neg_count / n_tokens
        return out


class NBPStackedClassifier(BaseEstimator):
    """Stacked NBP classifier:
    - 3 base models (LogReg+kw, LinearSVC+kw, CatBoost-text)
    - 8 rule features
    - Meta LogReg with frozen weights from training
    Predict returns 0/1 binary using stored threshold τ.
    """
    def __init__(self):
        self._fitted = False
        # Will be set in fit
        self.base_A = None
        self.base_F = None
        self.base_E = None
        self.meta = None
        self.threshold = 0.0
        # z-score reference (mean, std) of OOF arrays
        self.z_A = (0.0, 1.0)
        self.z_F = (0.0, 1.0)
        self.z_E = (0.0, 1.0)
        self.feature_set = 'A+F+E+rules'  # default

    def fit(self, X, y, oof_A, oof_F, oof_E, meta_C=0.5, threshold=None):
        """Fit base models on full data + meta on OOF features."""
        from catboost import CatBoostClassifier, Pool
        # Base A
        self.base_A = Pipeline([('features', FeatureUnion([
            ('w12', TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, lowercase=True)),
            ('c35', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, sublinear_tf=True, lowercase=True)),
            ('kw',  KeywordCountTransformer(NBP_KEYWORDS_EXTENDED, DKP_DISCRIMINATORS)),
        ])), ('clf', LogisticRegression(C=5.0, class_weight='balanced', solver='liblinear', max_iter=2000, random_state=42))])
        self.base_A.fit(X, y)
        # Base F
        self.base_F = Pipeline([('features', FeatureUnion([
            ('w12', TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, lowercase=True)),
            ('c35', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, sublinear_tf=True, lowercase=True)),
            ('kw',  KeywordCountTransformer(NBP_KEYWORDS_EXTENDED, DKP_DISCRIMINATORS)),
        ])), ('clf', LinearSVC(C=2.0, class_weight='balanced', max_iter=10000, dual='auto', random_state=42))])
        self.base_F.fit(X, y)
        # Base E (CatBoost native text)
        self.base_E = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=4,
            text_features=[0], auto_class_weights='Balanced',
            random_seed=42, verbose=0, thread_count=-1, allow_writing_files=False)
        df = pd.DataFrame({'text': X.values if hasattr(X, 'values') else X})
        self.base_E.fit(Pool(df, label=y.values if hasattr(y, 'values') else y, text_features=[0]))

        # z-score refs from OOF
        self.z_A = (oof_A.mean(), oof_A.std() + 1e-9)
        self.z_F = (oof_F.mean(), oof_F.std() + 1e-9)
        self.z_E = (oof_E.mean(), oof_E.std() + 1e-9)
        rules = rule_features_extended(X.values if hasattr(X, 'values') else X)
        X_meta = np.column_stack([
            (oof_A - self.z_A[0]) / self.z_A[1],
            (oof_F - self.z_F[0]) / self.z_F[1],
            (oof_E - self.z_E[0]) / self.z_E[1],
            rules,
        ])
        y_arr = y.values if hasattr(y, 'values') else y
        self.meta = LogisticRegression(C=meta_C, class_weight='balanced', solver='liblinear', random_state=42)
        self.meta.fit(X_meta, y_arr)
        self.threshold = float(threshold) if threshold is not None else 0.0
        self._fitted = True
        return self

    def _meta_features(self, X):
        from catboost import Pool
        sc_A = self.base_A.decision_function(X)
        sc_F = self.base_F.decision_function(X)
        df = pd.DataFrame({'text': X.values if hasattr(X, 'values') else X})
        sc_E = self.base_E.predict_proba(Pool(df, text_features=[0]))[:, 1]
        rules = rule_features_extended(X.values if hasattr(X, 'values') else X)
        return np.column_stack([
            (sc_A - self.z_A[0]) / self.z_A[1],
            (sc_F - self.z_F[0]) / self.z_F[1],
            (sc_E - self.z_E[0]) / self.z_E[1],
            rules,
        ])

    def decision_function(self, X):
        return self.meta.decision_function(self._meta_features(X))

    def predict(self, X):
        return (self.decision_function(X) >= self.threshold).astype(int)
