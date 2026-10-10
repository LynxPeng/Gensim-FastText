# Gensim 4.4.0 FastText 核心代码

日期：2026-10-09。以下代码是当前环境对应源文件的**原样连续摘录**，没有用教学伪代码替换实现。这里只输出核心调用链；完整源文件、许可证和 SHA-256 见末尾。代码块之间省略的外围逻辑不属于可独立运行模块。

对比结论：[Gensim_FastText与论文工程差异.md](<C:/Projects/LLM/Gensim/reports/Gensim_FastText与论文工程差异.md>)。本次数值核查：[audit.json](<C:/Projects/LLM/Gensim/artifacts/reviews/fasttext_paper_engineering/audit.json>)。

## 参数与训练调用链

`FastText → Word2Vec.build_vocab/train → _do_train_job / _do_train_epoch → fasttext_inner / fasttext_corpusfile → SG/CBOW × NS/HS → adjust_vectors`。

训练修改 `vectors_vocab`（词输入参数）、`vectors_ngrams`（字符桶输入参数）和 `syn1neg`／`syn1`（词输出参数）；`wv.vectors` 是查询使用的合成矩阵。输出层的行对应词或词树节点。外部标签分类器属于项目代码。

## 代码索引

| 编号 | 内容 | 源文件与行号 |
| --- | --- | --- |
| C01 | 构造函数和默认值 | [fasttext.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext.py:274>)，274–279 |
| C02 | 词 n-gram 限制、参数表与 Word2Vec 继承 | [fasttext.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext.py:418>)，418–442 |
| C03 | 训练入口：文件流与 iterable 分派 | [fasttext.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext.py:499>)，499–546 |
| C04 | UTF-8 字符 n-gram 提取 | [fasttext_inner.pyx](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext_inner.pyx:677>)，677–720 |
| C05 | FNV-1a 字节哈希 | [fasttext_inner.pyx](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext_inner.pyx:619>)，619–641 |
| C06 | 字符 n-gram 分桶入口 | [fasttext.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext.py:1326>)，1326–1349 |
| C07 | 训练批次的词表过滤与子词缓存绑定 | [fasttext_inner.pyx](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext_inner.pyx:470>)，470–510 |
| C08 | 训练窗口和 SG／CBOW、HS／NS 分派 | [fasttext_inner.pyx](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext_inner.pyx:513>)，513–569 |
| C09 | Skip-gram + Negative Sampling 完整核心函数 | [fasttext_inner.pyx](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext_inner.pyx:75>)，75–145 |
| C10 | Skip-gram + Hierarchical Softmax 完整核心函数 | [fasttext_inner.pyx](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext_inner.pyx:148>)，148–221 |
| C11 | CBOW + Negative Sampling 完整核心函数 | [fasttext_inner.pyx](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext_inner.pyx:224>)，224–306 |
| C12 | CBOW + Hierarchical Softmax 完整核心函数 | [fasttext_inner.pyx](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext_inner.pyx:309>)，309–373 |
| C13 | 负采样累计分布 | [word2vec.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/word2vec.py:824>)，824–851 |
| C14 | IV／OOV 向量查询 | [fasttext.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext.py:1085>)，1085–1130 |
| C15 | 稠密参数分配 | [fasttext.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext.py:1149>)，1149–1161 |
| C16 | 导出词向量合成与字符桶缓存 | [fasttext.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext.py:1191>)，1191–1227 |
| C17 | 句向量接口 | [fasttext.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext.py:1132>)，1132–1147 |
| C18 | corpus_file 复用相同训练内核 | [fasttext_corpusfile.pyx](<C:/Projects/LLM/Gensim/artifacts/reviews/fasttext_paper_engineering/sources/fasttext_corpusfile.pyx:18>)，18–28 |
| C19 | corpus_file 的 SG 更新与学习率调度 | [fasttext_corpusfile.pyx](<C:/Projects/LLM/Gensim/artifacts/reviews/fasttext_paper_engineering/sources/fasttext_corpusfile.pyx:158>)，158–187 |
| P01 | 项目表示模型训练 | [models.py](<C:/Projects/LLM/Gensim/gensim_lab/models.py:50>)，50–65 |
| P02 | 项目词向量池化与 L2 单位化 | [models.py](<C:/Projects/LLM/Gensim/gensim_lab/models.py:95>)，95–125 |
| P03 | 项目外部 SVM 分类器 | [models.py](<C:/Projects/LLM/Gensim/gensim_lab/models.py:155>)，155–173 |
| P04 | 论文复现路线的文档池化 | [paper_worker.py](<C:/Projects/LLM/Gensim/gensim_lab/paper_worker.py:48>)，48–73 |
| N01 | 原生模型平均隐藏层与梯度更新 | [model.cc](<C:/Projects/LLM/Gensim/references/sources/fastText-0.9.2/src/model.cc:42>)，42–91 |
| N02 | 原生负采样分布与重抽样 | [loss.cc](<C:/Projects/LLM/Gensim/references/sources/fastText-0.9.2/src/loss.cc:145>)，145–190 |
| N03 | 原生输入词 ID 与 EOS 特判 | [dictionary.cc](<C:/Projects/LLM/Gensim/references/sources/fastText-0.9.2/src/dictionary.cc:196>)，196–205 |
| N04 | 原生 SG／CBOW 的输入特征构建 | [fasttext.cc](<C:/Projects/LLM/Gensim/references/sources/fastText-0.9.2/src/fasttext.cc:387>)，387–421 |
| N05 | 原生梯度归一化开关 | [fasttext.cc](<C:/Projects/LLM/Gensim/references/sources/fastText-0.9.2/src/fasttext.cc:762>)，762–766 |


## C01 构造函数和默认值


来源：[fasttext.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext.py:274>)，原文件第 274–279 行。


```python
    def __init__(self, sentences=None, corpus_file=None, sg=0, hs=0, vector_size=100, alpha=0.025,
                 window=5, min_count=5,
                 max_vocab_size=None, word_ngrams=1, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, epochs=5, null_word=0, min_n=3, max_n=6,
                 sorted_vocab=1, bucket=2000000, trim_rule=None, batch_words=MAX_WORDS_IN_BATCH, callbacks=(),
                 max_final_vocab=None, shrink_windows=True,):
```


## C02 词 n-gram 限制、参数表与 Word2Vec 继承


来源：[fasttext.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext.py:418>)，原文件第 418–442 行。


```python
        """
        self.load = utils.call_on_class_only
        self.load_fasttext_format = utils.call_on_class_only
        self.callbacks = callbacks
        if word_ngrams != 1:
            raise NotImplementedError("Gensim's FastText implementation does not yet support word_ngrams != 1.")
        self.word_ngrams = word_ngrams
        if max_n < min_n:
            # with no eligible char-ngram lengths, no buckets need be allocated
            bucket = 0

        self.wv = FastTextKeyedVectors(vector_size, min_n, max_n, bucket)
        # EXPERIMENTAL lockf feature; create minimal no-op lockf arrays (1 element of 1.0)
        # advanced users should directly resize/adjust as desired after any vocab growth
        self.wv.vectors_vocab_lockf = ones(1, dtype=REAL)
        self.wv.vectors_ngrams_lockf = ones(1, dtype=REAL)

        super(FastText, self).__init__(
            sentences=sentences, corpus_file=corpus_file, workers=workers, vector_size=vector_size, epochs=epochs,
            callbacks=callbacks, batch_words=batch_words, trim_rule=trim_rule, sg=sg, alpha=alpha, window=window,
            max_vocab_size=max_vocab_size, max_final_vocab=max_final_vocab,
            min_count=min_count, sample=sample, sorted_vocab=sorted_vocab,
            null_word=null_word, ns_exponent=ns_exponent, hashfxn=hashfxn,
            seed=seed, hs=hs, negative=negative, cbow_mean=cbow_mean,
            min_alpha=min_alpha, shrink_windows=shrink_windows)
```


## C03 训练入口：文件流与 iterable 分派


来源：[fasttext.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext.py:499>)，原文件第 499–546 行。


```python
    def _do_train_epoch(
            self, corpus_file, thread_id, offset, cython_vocab, thread_private_mem, cur_epoch,
            total_examples=None, total_words=None, **kwargs,
        ):
        work, neu1 = thread_private_mem

        if self.sg:
            examples, tally, raw_tally = train_epoch_sg(
                self, corpus_file, offset, cython_vocab, cur_epoch, total_examples, total_words, work, neu1,
            )
        else:
            examples, tally, raw_tally = train_epoch_cbow(
                self, corpus_file, offset, cython_vocab, cur_epoch, total_examples, total_words, work, neu1,
            )

        return examples, tally, raw_tally

    def _do_train_job(self, sentences, alpha, inits):
        """Train a single batch of sentences. Return 2-tuple `(effective word count after
        ignoring unknown words and sentence length trimming, total word count)`.

        Parameters
        ----------
        sentences : iterable of list of str
            Can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        alpha : float
            The current learning rate.
        inits : tuple of (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Each worker's private work memory.

        Returns
        -------
        (int, int)
            Tuple of (effective word count after ignoring unknown words and sentence length trimming, total word count)

        """
        work, neu1 = inits
        tally = train_batch_any(self, sentences, alpha, work, neu1)

        return tally, self._raw_word_count(sentences)

    @deprecated(
        "Gensim 4.0.0 implemented internal optimizations that make calls to init_sims() unnecessary. "
        "init_sims() is now obsoleted and will be completely removed in future versions. "
        "See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4"
```


## C04 UTF-8 字符 n-gram 提取


来源：[fasttext_inner.pyx](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext_inner.pyx:677>)，原文件第 677–720 行。


```cython
cpdef compute_ngrams_bytes(word, unsigned int min_n, unsigned int max_n):
    """Computes ngrams for a word.

    Ported from the original FB implementation.

    Parameters
    ----------
    word : str
        A unicode string.
    min_n : unsigned int
        The minimum ngram length.
    max_n : unsigned int
        The maximum ngram length.

    Returns:
    --------
    list of str
        A list of ngrams, where each ngram is a list of **bytes**.

    See Also
    --------
    `Original implementation <https://github.com/facebookresearch/fastText/blob/7842495a4d64c7a3bb4339d45d6e64321d002ed8/src/dictionary.cc#L172>`__

    """
    cdef bytes utf8_word = ('<%s>' % word).encode("utf-8")
    cdef const unsigned char *bytez = utf8_word
    cdef size_t num_bytes = len(utf8_word)
    cdef size_t j, i, n

    ngrams = []
    for i in range(num_bytes):
        if bytez[i] & _MB_MASK == _MB_START:
            continue

        j, n = i, 1
        while j < num_bytes and n <= max_n:
            j += 1
            while j < num_bytes and (bytez[j] & _MB_MASK) == _MB_START:
                j += 1
            if n >= min_n and not (n == 1 and (i == 0 or j == num_bytes)):
                ngram = bytes(bytez[i:j])
                ngrams.append(ngram)
            n += 1
    return ngrams
```


## C05 FNV-1a 字节哈希


来源：[fasttext_inner.pyx](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext_inner.pyx:619>)，原文件第 619–641 行。


```cython
cpdef ft_hash_bytes(bytes bytez):
    """Calculate hash based on `bytez`.
    Reproduce `hash method from Facebook fastText implementation
    <https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc>`_.

    Parameters
    ----------
    bytez : bytes
        The string whose hash needs to be calculated, encoded as UTF-8.

    Returns
    -------
    unsigned int
        The hash of the string.

    """
    cdef np.uint32_t h = 2166136261
    cdef char b

    for b in bytez:
        h = h ^ <np.uint32_t>(<np.int8_t>b)
        h = h * 16777619
    return h
```


## C06 字符 n-gram 分桶入口


来源：[fasttext.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext.py:1326>)，原文件第 1326–1349 行。


```python
def ft_ngram_hashes(word, minn, maxn, num_buckets):
    """Calculate the ngrams of the word and hash them.

    Parameters
    ----------
    word : str
        The word to calculate ngram hashes for.
    minn : int
        Minimum ngram length
    maxn : int
        Maximum ngram length
    num_buckets : int
        The number of buckets

    Returns
    -------
        A list of hashes (integers), one per each detected ngram.

    """
    encoded_ngrams = compute_ngrams_bytes(word, minn, maxn)
    hashes = [ft_hash_bytes(n) % num_buckets for n in encoded_ngrams]
    return hashes


```


## C07 训练批次的词表过滤与子词缓存绑定


来源：[fasttext_inner.pyx](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext_inner.pyx:470>)，原文件第 470–510 行。


```cython

    if c.sample:
        vocab_sample_ints = <np.uint32_t *>np.PyArray_DATA(wv.expandos['sample_int'])
    if c.hs:
        vocab_codes = wv.expandos['code']
        vocab_points = wv.expandos['point']
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word_index = wv.key_to_index.get(token, None)
            if word_index is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if c.sample and vocab_sample_ints[word_index] < random_int32(&c.next_random):
                continue
            c.indexes[effective_words] = word_index

            if wv.bucket:
                c.subwords_idx_len[effective_words] = <int>(len(buckets_word[word_index]))
                c.subwords_idx[effective_words] = <np.uint32_t *>np.PyArray_DATA(buckets_word[word_index])
            else:
                c.subwords_idx_len[effective_words] = 0

            if c.hs:
                c.codelens[effective_words] = <int>len(vocab_codes[word_index])
                c.codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(vocab_codes[word_index])
                c.points[effective_words] = <np.uint32_t *>np.PyArray_DATA(vocab_points[word_index])

            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        effective_sentences += 1
        c.sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break

    return effective_words, effective_sentences
```


## C08 训练窗口和 SG／CBOW、HS／NS 分派


来源：[fasttext_inner.pyx](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext_inner.pyx:513>)，原文件第 513–569 行。


```cython
cdef void fasttext_train_any(FastTextConfig *c, int num_sentences) noexcept nogil:
    """Performs training on a fully initialized and populated configuration.

    Parameters
    ----------
    c : FastTextConfig *
        A pointer to the configuration struct.
    num_sentences : int
        The number of sentences to train.

    """
    cdef:
        int sent_idx
        int sentence_start
        int sentence_end
        int i
        int window_start
        int window_end
        int j

    for sent_idx in range(num_sentences):
        sentence_start = c.sentence_idx[sent_idx]
        sentence_end = c.sentence_idx[sent_idx + 1]
        for i in range(sentence_start, sentence_end):
            #
            # Determine window boundaries, making sure we don't leak into
            # adjacent sentences.
            #
            window_start = i - c.window + c.reduced_windows[i]
            if window_start < sentence_start:
                window_start = sentence_start
            window_end = i + c.window + 1 - c.reduced_windows[i]
            if window_end > sentence_end:
                window_end = sentence_end

            #
            # TODO: why can't I use min/max here?  I get a segfault.
            #
            # window_start = max(sentence_start, i - c.window + c.reduced_windows[i])
            # window_end = min(sentence_end, i + c.window + 1 - c.reduced_windows[i])
            #
            if c.sg == 0:
                if c.hs:
                    fasttext_fast_sentence_cbow_hs(c, i, window_start, window_end)
                if c.negative:
                    fasttext_fast_sentence_cbow_neg(c, i, window_start, window_end)
            else:
                for j in range(window_start, window_end):
                    if j == i:
                        # no reason to train a center word as predicting itself
                        continue
                    if c.hs:
                        fasttext_fast_sentence_sg_hs(c, i, j)
                    if c.negative:
                        fasttext_fast_sentence_sg_neg(c, i, j)


```


## C09 Skip-gram + Negative Sampling 完整核心函数


来源：[fasttext_inner.pyx](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext_inner.pyx:75>)，原文件第 75–145 行。


```cython
cdef void fasttext_fast_sentence_sg_neg(FastTextConfig *c, int i, int j) noexcept nogil:
    """Perform skipgram training with negative sampling.

    Parameters
    ----------
    c : FastTextConfig *
        A pointer to a fully initialized and populated struct.
    i : int
        The index of the word at the center of the current window.  This is
        referred to as word2 in some parts of the implementation.
    j : int
        The index of another word inside the window.  This is referred to as
        word in some parts of the implementation.

    Notes
    -----
    Modifies c.next_random as a side-effect.

    """

    cdef:
        np.uint32_t word_index = c.indexes[j]
        np.uint32_t word2_index = c.indexes[i]
        np.uint32_t *subwords_index = c.subwords_idx[i]
        np.uint32_t subwords_len = c.subwords_idx_len[i]

    cdef long long row1 = word2_index * c.size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot
    cdef np.uint32_t target_index
    cdef int d

    memset(c.work, 0, c.size * cython.sizeof(REAL_t))
    memset(c.neu1, 0, c.size * cython.sizeof(REAL_t))

    scopy(&c.size, &c.syn0_vocab[row1], &ONE, c.neu1, &ONE)

    #
    # Avoid division by zero.
    #
    cdef REAL_t norm_factor
    if subwords_len:
        for d in range(subwords_len):
            our_saxpy(&c.size, &ONEF, &c.syn0_ngrams[subwords_index[d] * c.size], &ONE, c.neu1, &ONE)
        norm_factor = ONEF / subwords_len
        sscal(&c.size, &norm_factor, c.neu1, &ONE)

    for d in range(c.negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(
                c.cum_table, (c.next_random >> 16) % c.cum_table[c.cum_table_len-1], 0, c.cum_table_len)
            c.next_random = (c.next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * c.size
        f_dot = our_dot(&c.size, c.neu1, &ONE, &c.syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * c.alpha
        our_saxpy(&c.size, &g, &c.syn1neg[row2], &ONE, c.work, &ONE)
        our_saxpy(&c.size, &g, c.neu1, &ONE, &c.syn1neg[row2], &ONE)
    our_saxpy(&c.size, &c.vocab_lockf[word2_index % c.vocab_lockf_len], c.work, &ONE, &c.syn0_vocab[row1], &ONE)
    for d in range(subwords_len):
        our_saxpy(&c.size, &c.ngrams_lockf[subwords_index[d] % c.ngrams_lockf_len],
                  c.work, &ONE, &c.syn0_ngrams[subwords_index[d]*c.size], &ONE)
```


## C10 Skip-gram + Hierarchical Softmax 完整核心函数


来源：[fasttext_inner.pyx](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext_inner.pyx:148>)，原文件第 148–221 行。


```cython
cdef void fasttext_fast_sentence_sg_hs(FastTextConfig *c, int i, int j) noexcept nogil:
    """Perform skipgram training with hierarchical sampling.

    Parameters
    ----------
    c : FastTextConfig *
        A pointer to a fully initialized and populated struct.
    i : int
        The index of the word at the center of the current window.  This is
        referred to as word2 in some parts of the implementation.
    j : int
        The index of another word inside the window.  This is referred to as
        word in some parts of the implementation.

    """
    cdef:
        np.uint32_t *word_point = c.points[j]
        np.uint8_t *word_code = c.codes[j]
        int codelen = c.codelens[j]
        np.uint32_t word2_index = c.indexes[i]
        np.uint32_t *subwords_index = c.subwords_idx[i]
        np.uint32_t subwords_len = c.subwords_idx_len[i]

    #
    # b : long long
    #   iteration variable
    # row1 : long long
    #   Offset for word2 (!!) into the syn0_vocab array
    # row2 : long long
    #   Another offset into the syn0_vocab array
    # f : REAL_t
    #   ?
    # f_dot : REAL_t
    #   Dot product result
    # g : REAL_t
    #   ?
    #
    cdef long long b
    cdef long long row1 = word2_index * c.size, row2
    cdef REAL_t f, g, f_dot

    memset(c.work, 0, c.size * cython.sizeof(REAL_t))
    memset(c.neu1, 0, c.size * cython.sizeof(REAL_t))

    scopy(&c.size, &c.syn0_vocab[row1], &ONE, c.neu1, &ONE)

    #
    # Avoid division by zero.
    #
    cdef REAL_t norm_factor
    if subwords_len:
        for d in range(subwords_len):
            row2 = subwords_index[d] * c.size
            our_saxpy(&c.size, &ONEF, &c.syn0_ngrams[row2], &ONE, c.neu1, &ONE)
        norm_factor = ONEF / subwords_len
        sscal(&c.size, &norm_factor, c.neu1, &ONE)

    for b in range(codelen):
        row2 = word_point[b] * c.size
        f_dot = our_dot(&c.size, c.neu1, &ONE, &c.syn1[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * c.alpha

        our_saxpy(&c.size, &g, &c.syn1[row2], &ONE, c.work, &ONE)
        our_saxpy(&c.size, &g, c.neu1, &ONE, &c.syn1[row2], &ONE)

    our_saxpy(&c.size, &c.vocab_lockf[word2_index % c.vocab_lockf_len], c.work, &ONE, &c.syn0_vocab[row1], &ONE)
    for d in range(subwords_len):
        row2 = subwords_index[d] * c.size
        our_saxpy(
            &c.size, &c.ngrams_lockf[subwords_index[d] % c.ngrams_lockf_len], c.work, &ONE,
            &c.syn0_ngrams[row2], &ONE)
```


## C11 CBOW + Negative Sampling 完整核心函数


来源：[fasttext_inner.pyx](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext_inner.pyx:224>)，原文件第 224–306 行。


```cython
cdef void fasttext_fast_sentence_cbow_neg(FastTextConfig *c, int i, int j, int k) noexcept nogil:
    """Perform CBOW training with negative sampling.

    Parameters
    ----------
    c : FastTextConfig *
        A pointer to a fully initialized and populated struct.
    i : int
        The index of a word inside the current window.
    j : int
        The start of the current window.
    k : int
        The end of the current window.  Essentially, j <= i < k.

    Notes
    -----
    Modifies c.next_random as a side-effect.

    """

    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count = 1.0, label, f_dot
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    word_index = c.indexes[i]

    memset(c.neu1, 0, c.size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        count += ONEF
        our_saxpy(&c.size, &ONEF, &c.syn0_vocab[c.indexes[m] * c.size], &ONE, c.neu1, &ONE)
        for d in range(c.subwords_idx_len[m]):
            count += ONEF
            our_saxpy(&c.size, &ONEF, &c.syn0_ngrams[c.subwords_idx[m][d] * c.size], &ONE, c.neu1, &ONE)

    if count > (<REAL_t>0.5):
        inv_count = ONEF / count
    if c.cbow_mean:
        sscal(&c.size, &inv_count, c.neu1, &ONE)

    memset(c.work, 0, c.size * cython.sizeof(REAL_t))

    for d in range(c.negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(c.cum_table, (c.next_random >> 16) % c.cum_table[c.cum_table_len-1], 0, c.cum_table_len)
            c.next_random = (c.next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * c.size
        f_dot = our_dot(&c.size, c.neu1, &ONE, &c.syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP:
            f = 0.0
        elif f_dot >= MAX_EXP:
            f = 1.0
        else:
            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * c.alpha

        our_saxpy(&c.size, &g, &c.syn1neg[row2], &ONE, c.work, &ONE)
        our_saxpy(&c.size, &g, c.neu1, &ONE, &c.syn1neg[row2], &ONE)

    if not c.cbow_mean:  # divide error over summed window vectors
        sscal(&c.size, &inv_count, c.work, &ONE)

    for m in range(j,k):
        if m == i:
            continue
        our_saxpy(
            &c.size, &c.vocab_lockf[c.indexes[m] % c.vocab_lockf_len], c.work, &ONE,
            &c.syn0_vocab[c.indexes[m]*c.size], &ONE)
        for d in range(c.subwords_idx_len[m]):
            our_saxpy(
                &c.size, &c.ngrams_lockf[c.subwords_idx[m][d] % c.ngrams_lockf_len], c.work, &ONE,
                &c.syn0_ngrams[c.subwords_idx[m][d]*c.size], &ONE)
```


## C12 CBOW + Hierarchical Softmax 完整核心函数


来源：[fasttext_inner.pyx](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext_inner.pyx:309>)，原文件第 309–373 行。


```cython
cdef void fasttext_fast_sentence_cbow_hs(FastTextConfig *c, int i, int j, int k) noexcept nogil:
    """Perform CBOW training with hierarchical sampling.

    Parameters
    ----------
    c : FastTextConfig *
        A pointer to a fully initialized and populated struct.
    i : int
        The index of a word inside the current window.
    j : int
        The start of the current window.
    k : int
        The end of the current window.  Essentially, j <= i < k.

    """

    cdef:
        np.uint32_t *word_point = c.points[i]
        np.uint8_t *word_code = c.codes[i]

    cdef long long b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count = 1.0, f_dot
    cdef int m

    memset(c.neu1, 0, c.size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        count += ONEF
        our_saxpy(&c.size, &ONEF, &c.syn0_vocab[c.indexes[m] * c.size], &ONE, c.neu1, &ONE)
        for d in range(c.subwords_idx_len[m]):
            count += ONEF
            our_saxpy(&c.size, &ONEF, &c.syn0_ngrams[c.subwords_idx[m][d] * c.size], &ONE, c.neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF / count
    if c.cbow_mean:
        sscal(&c.size, &inv_count, c.neu1, &ONE)

    memset(c.work, 0, c.size * cython.sizeof(REAL_t))
    for b in range(c.codelens[i]):
        row2 = word_point[b] * c.size
        f_dot = our_dot(&c.size, c.neu1, &ONE, &c.syn1[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * c.alpha

        our_saxpy(&c.size, &g, &c.syn1[row2], &ONE, c.work, &ONE)
        our_saxpy(&c.size, &g, c.neu1, &ONE, &c.syn1[row2], &ONE)

    if not c.cbow_mean:  # divide error over summed window vectors
        sscal(&c.size, &inv_count, c.work, &ONE)

    for m in range(j,k):
        if m == i:
            continue
        our_saxpy(
            &c.size, &c.vocab_lockf[c.indexes[m] % c.vocab_lockf_len], c.work, &ONE,
            &c.syn0_vocab[c.indexes[m]*c.size], &ONE)
        for d in range(c.subwords_idx_len[m]):
            our_saxpy(
                &c.size, &c.ngrams_lockf[c.subwords_idx[m][d] % c.ngrams_lockf_len], c.work, &ONE,
                &c.syn0_ngrams[c.subwords_idx[m][d]*c.size], &ONE)
```


## C13 负采样累计分布


来源：[word2vec.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/word2vec.py:824>)，原文件第 824–851 行。


```python

    def make_cum_table(self, domain=2**31 - 1):
        """Create a cumulative-distribution table using stored vocabulary word counts for
        drawing random words in the negative-sampling training routines.

        To draw a word index, choose a random integer up to the maximum value in the table (cum_table[-1]),
        then finding that integer's sorted insertion point (as if by `bisect_left` or `ndarray.searchsorted()`).
        That insertion point is the drawn index, coming up in proportion equal to the increment at that slot.

        """
        vocab_size = len(self.wv.index_to_key)
        self.cum_table = np.zeros(vocab_size, dtype=np.uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = 0.0
        for word_index in range(vocab_size):
            count = self.wv.get_vecattr(word_index, 'count')
            train_words_pow += count**float(self.ns_exponent)
        cumulative = 0.0
        for word_index in range(vocab_size):
            count = self.wv.get_vecattr(word_index, 'count')
            cumulative += count**float(self.ns_exponent)
            self.cum_table[word_index] = round(cumulative / train_words_pow * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain

    def prepare_weights(self, update=False):
        """Build tables and model weights based on final vocabulary settings."""
        # set initial input/projection and hidden weights
```


## C14 IV／OOV 向量查询


来源：[fasttext.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext.py:1085>)，原文件第 1085–1130 行。


```python
    def get_vector(self, word, norm=False):
        """Get `word` representations in vector space, as a 1D numpy array.

        Parameters
        ----------
        word : str
            Input word.
        norm : bool, optional
            If True, resulting vector will be L2-normalized (unit Euclidean length).

        Returns
        -------
        numpy.ndarray
            Vector representation of `word`.

        Raises
        ------
        KeyError
            If word and all its ngrams not in vocabulary.

        """
        if word in self.key_to_index:
            return super(FastTextKeyedVectors, self).get_vector(word, norm=norm)
        elif self.bucket == 0:
            raise KeyError('cannot calculate vector for OOV word without ngrams')
        else:
            word_vec = np.zeros(self.vectors_ngrams.shape[1], dtype=np.float32)
            ngram_weights = self.vectors_ngrams
            ngram_hashes = ft_ngram_hashes(word, self.min_n, self.max_n, self.bucket)
            if len(ngram_hashes) == 0:
                #
                # If it is impossible to extract _any_ ngrams from the input
                # word, then the best we can do is return a vector that points
                # to the origin.  The reference FB implementation does this,
                # too.
                #
                # https://github.com/RaRe-Technologies/gensim/issues/2402
                #
                logger.warning('could not extract any ngrams from %r, returning origin vector', word)
                return word_vec
            for nh in ngram_hashes:
                word_vec += ngram_weights[nh]
            if norm:
                return word_vec / np.linalg.norm(word_vec)
            else:
                return word_vec / len(ngram_hashes)
```


## C15 稠密参数分配


来源：[fasttext.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext.py:1149>)，原文件第 1149–1161 行。


```python
    def resize_vectors(self, seed=0):
        """Make underlying vectors match 'index_to_key' size; random-initialize any new rows."""

        vocab_shape = (len(self.index_to_key), self.vector_size)
        # Unlike in superclass, 'vectors_vocab' array is primary with 'vectors' derived from it & ngrams
        self.vectors_vocab = prep_vectors(vocab_shape, prior_vectors=self.vectors_vocab, seed=seed)
        ngrams_shape = (self.bucket, self.vector_size)
        self.vectors_ngrams = prep_vectors(ngrams_shape, prior_vectors=self.vectors_ngrams, seed=seed + 1)

        self.allocate_vecattrs()
        self.norms = None
        self.recalc_char_ngram_buckets()  # ensure new words have precalc buckets
        self.adjust_vectors()  # ensure `vectors` filled as well (though may be nonsense pre-training)
```


## C16 导出词向量合成与字符桶缓存


来源：[fasttext.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext.py:1191>)，原文件第 1191–1227 行。


```python
    def adjust_vectors(self):
        """Adjust the vectors for words in the vocabulary.

        The adjustment composes the trained full-word-token vectors with
        the vectors of the subword ngrams, matching the Facebook reference
        implementation behavior.

        """
        if self.bucket == 0:
            self.vectors = self.vectors_vocab  # no ngrams influence
            return

        self.vectors = self.vectors_vocab[:].copy()
        for i, _ in enumerate(self.index_to_key):
            ngram_buckets = self.buckets_word[i]
            for nh in ngram_buckets:
                self.vectors[i] += self.vectors_ngrams[nh]
            self.vectors[i] /= len(ngram_buckets) + 1

    def recalc_char_ngram_buckets(self):
        """
        Scan the vocabulary, calculate ngrams and their hashes, and cache the list of ngrams for each known word.

        """
        # TODO: evaluate if precaching even necessary, compared to recalculating as needed.
        if self.bucket == 0:
            self.buckets_word = [np.array([], dtype=np.uint32)] * len(self.index_to_key)
            return

        self.buckets_word = [None] * len(self.index_to_key)

        for i, word in enumerate(self.index_to_key):
            self.buckets_word[i] = np.array(
                ft_ngram_hashes(word, self.min_n, self.max_n, self.bucket),
                dtype=np.uint32,
           )

```


## C17 句向量接口


来源：[fasttext.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/fasttext.py:1132>)，原文件第 1132–1147 行。


```python
    def get_sentence_vector(self, sentence):
        """Get a single 1-D vector representation for a given `sentence`.
        This function is workalike of the official fasttext's get_sentence_vector().

        Parameters
        ----------
        sentence : list of (str or int)
            list of words specified by string or int ids.

        Returns
        -------
        numpy.ndarray
            1-D numpy array representation of the `sentence`.

        """
        return super(FastTextKeyedVectors, self).get_mean_vector(sentence)
```


## C18 corpus_file 复用相同训练内核


来源：[fasttext_corpusfile.pyx](<C:/Projects/LLM/Gensim/artifacts/reviews/fasttext_paper_engineering/sources/fasttext_corpusfile.pyx:18>)，原文件第 18–28 行。


```cython
from libcpp.string cimport string
from libcpp.vector cimport vector

from gensim.models.fasttext_inner cimport (
    fasttext_fast_sentence_sg_hs,
    fasttext_fast_sentence_sg_neg,
    fasttext_fast_sentence_cbow_hs,
    fasttext_fast_sentence_cbow_neg,
    init_ft_config,
    FastTextConfig
)
```


## C19 corpus_file 的 SG 更新与学习率调度


来源：[fasttext_corpusfile.pyx](<C:/Projects/LLM/Gensim/artifacts/reviews/fasttext_paper_engineering/sources/fasttext_corpusfile.pyx:158>)，原文件第 158–187 行。


```cython
            sentences = input_stream.next_batch()

            prepare_c_structures_for_batch(
                sentences, c.sample, c.hs, c.window, &total_words, &effective_words, &effective_sentences,
                &c.next_random, vocab.get_vocab_ptr(), c.sentence_idx, c.indexes, c.codelens,
                c.codes, c.points, c.reduced_windows, c.subwords_idx_len, c.subwords_idx, shrink_windows)

            for sent_idx in range(effective_sentences):
                idx_start = c.sentence_idx[sent_idx]
                idx_end = c.sentence_idx[sent_idx + 1]
                for i in range(idx_start, idx_end):
                    j = i - c.window + c.reduced_windows[i]
                    if j < idx_start:
                        j = idx_start
                    k = i + c.window + 1 - c.reduced_windows[i]
                    if k > idx_end:
                        k = idx_end
                    for j in range(j, k):
                        if j == i:
                            continue
                        if c.hs:
                            fasttext_fast_sentence_sg_hs(&c, i, j)
                        if c.negative:
                            fasttext_fast_sentence_sg_neg(&c, i, j)

            total_sentences += sentences.size()
            total_effective_words += effective_words

            c.alpha = get_next_alpha(start_alpha, end_alpha, total_sentences, total_words,
                                     expected_examples, expected_words, cur_epoch, num_epochs)
```


## P01 项目表示模型训练


来源：[models.py](<C:/Projects/LLM/Gensim/gensim_lab/models.py:50>)，原文件第 50–65 行。


```python
        base = {k: p[k] for k in ["vector_size", "window", "min_count", "epochs", "negative", "hs", "sample", "alpha", "min_alpha", "workers", "batch_words"]}
        base["seed"] = self.seed
        if self.method == "lsi":
            topics = min(p["vector_size"], len(self.dictionary), max(1, len(docs) - 1))
            self.model = LsiModel(self.tfidf[bows], id2word=self.dictionary, num_topics=topics, random_seed=self.seed)
        elif self.method == "word2vec":
            self.model = Word2Vec(sentences=SentenceCorpus(docs), sg=p["sg"], **base)
        elif self.method == "fasttext":
            self.model = FastText(sentences=SentenceCorpus(docs), sg=p["sg"], min_n=p["min_n"], max_n=p["max_n"], bucket=p["bucket"], **base)
        elif self.method == "doc2vec":
            tagged = [TaggedDocument(d["tokens"], [d["id"]]) for d in docs]
            self.model = Doc2Vec(documents=tagged, dm=0, **base)
        elif self.method != "tfidf":
            raise ValueError(self.method)
        self.max_idf = max(self.tfidf.idfs.values(), default=1.0)
        self.fit_ids = [d["id"] for d in docs]
```


## P02 项目词向量池化与 L2 单位化


来源：[models.py](<C:/Projects/LLM/Gensim/gensim_lab/models.py:95>)，原文件第 95–125 行。


```python
        else:
            for d in docs:
                total += len(d["tokens"])
                result = np.zeros(self.parameters["vector_size"], dtype=np.float32)
                weight_sum = 0.0
                counts = collections.Counter(d["tokens"])
                for word, count in counts.items():
                    unseen = word not in vocabulary
                    oov += count if unseen else 0
                    try:
                        v = self.model.wv.get_vector(word)
                        if not np.all(np.isfinite(v)) or not np.any(v):
                            continue
                    except KeyError:
                        continue
                    valid_oov += count if unseen else 0
                    weight = float(count)
                    if self.parameters["pooling"] == "tfidf":
                        token_id = self.dictionary.token2id.get(word)
                        weight *= self.tfidf.idfs.get(token_id, self.max_idf)
                    result += weight * v
                    weight_sum += weight
                if weight_sum:
                    result /= weight_sum
                else:
                    zero_docs += 1
                vectors.append(result)
            matrix = np.asarray(vectors, dtype=np.float32)
        matrix = normalize(matrix, norm="l2", copy=False)
        coverage = {"tokens": total, "oov_tokens": oov, "oov_rate": oov / total if total else 0.0, "oov_vector_coverage": valid_oov / oov if oov else 1.0, "zero_documents": int(zero_docs)}
        if not np.all(np.isfinite(matrix.data if sparse.issparse(matrix) else matrix)):
```


## P03 项目外部 SVM 分类器


来源：[models.py](<C:/Projects/LLM/Gensim/gensim_lab/models.py:155>)，原文件第 155–173 行。


```python
            path.write_text("".join(f"__label__p{d['page_label']} {' '.join(d['tokens'])}\n" for d in train), encoding="utf-8")
            self.native = fasttext.train_supervised(input=str(path), seed=self.spec["seed"], verbose=0, **self.spec["native"])
        elif self.method == "majority":
            self.classifier = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train), 1)), y)
        else:
            self.representation = Representation(self.method, self.spec["embedding"], self.spec["dictionary"], self.spec["seed"]).fit(train)
            x, _ = self.representation.transform(train)
            xv, _ = self.representation.transform(validation)
            best = None
            for c in self.spec["svm_c"]:
                clf = LinearSVC(C=c, class_weight="balanced", random_state=self.spec["seed"], dual="auto", max_iter=20000)
                clf.fit(x, y)
                score = metrics([d["page_label"] for d in validation], clf.predict(xv), self.labels)["macro_f1"]
                tuning.append({"C": c, "validation_macro_f1": score})
                if best is None or score > best[0]:
                    best = score, clf
            self.classifier = best[1]
        self.fit_seconds = time.perf_counter() - fit_start
        self.tuning = tuning
```


## P04 论文复现路线的文档池化


来源：[paper_worker.py](<C:/Projects/LLM/Gensim/gensim_lab/paper_worker.py:48>)，原文件第 48–73 行。


```python
    counts = collections.Counter(words)
    vector, accepted = np.zeros(dimension, np.float32), 0
    missing = 0
    for token, frequency in counts.items():
        index = model.wv.key_to_index.get(token)
        if index is not None:
            value = model.wv.vectors[index]
        else:
            missing += frequency
            try:
                value = cache[token] if cache is not None and token in cache else model.wv.get_vector(token)
                if cache is not None and len(cache) < 100000:
                    cache[token] = value
            except KeyError:
                continue
        vector += value * frequency
        accepted += frequency
    if accepted:
        vector /= accepted
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
    if not np.isfinite(vector).all():
        raise ValueError("Non-finite document vector")
    return vector, missing, not bool(accepted)

```


## N01 原生模型平均隐藏层与梯度更新


来源：[model.cc](<C:/Projects/LLM/Gensim/references/sources/fastText-0.9.2/src/model.cc:42>)，原文件第 42–91 行。


```cpp
void Model::computeHidden(const std::vector<int32_t>& input, State& state)
    const {
  Vector& hidden = state.hidden;
  hidden.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden.addRow(*wi_, *it);
  }
  hidden.mul(1.0 / input.size());
}

void Model::predict(
    const std::vector<int32_t>& input,
    int32_t k,
    real threshold,
    Predictions& heap,
    State& state) const {
  if (k == Model::kUnlimitedPredictions) {
    k = wo_->size(0); // output size
  } else if (k <= 0) {
    throw std::invalid_argument("k needs to be 1 or higher!");
  }
  heap.reserve(k + 1);
  computeHidden(input, state);

  loss_->predict(k, threshold, heap, state);
}

void Model::update(
    const std::vector<int32_t>& input,
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    real lr,
    State& state) {
  if (input.size() == 0) {
    return;
  }
  computeHidden(input, state);

  Vector& grad = state.grad;
  grad.zero();
  real lossValue = loss_->forward(targets, targetIndex, state, lr, true);
  state.incrementNExamples(lossValue);

  if (normalizeGradient_) {
    grad.mul(1.0 / input.size());
  }
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_->addVectorToRow(grad, *it, 1.0);
  }
}
```


## N02 原生负采样分布与重抽样


来源：[loss.cc](<C:/Projects/LLM/Gensim/references/sources/fastText-0.9.2/src/loss.cc:145>)，原文件第 145–190 行。


```cpp
NegativeSamplingLoss::NegativeSamplingLoss(
    std::shared_ptr<Matrix>& wo,
    int neg,
    const std::vector<int64_t>& targetCounts)
    : BinaryLogisticLoss(wo), neg_(neg), negatives_(), uniform_() {
  real z = 0.0;
  for (size_t i = 0; i < targetCounts.size(); i++) {
    z += pow(targetCounts[i], 0.5);
  }
  for (size_t i = 0; i < targetCounts.size(); i++) {
    real c = pow(targetCounts[i], 0.5);
    for (size_t j = 0; j < c * NegativeSamplingLoss::NEGATIVE_TABLE_SIZE / z;
         j++) {
      negatives_.push_back(i);
    }
  }
  uniform_ = std::uniform_int_distribution<size_t>(0, negatives_.size() - 1);
}

real NegativeSamplingLoss::forward(
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    Model::State& state,
    real lr,
    bool backprop) {
  assert(targetIndex >= 0);
  assert(targetIndex < targets.size());
  int32_t target = targets[targetIndex];
  real loss = binaryLogistic(target, state, true, lr, backprop);

  for (int32_t n = 0; n < neg_; n++) {
    auto negativeTarget = getNegative(target, state.rng);
    loss += binaryLogistic(negativeTarget, state, false, lr, backprop);
  }
  return loss;
}

int32_t NegativeSamplingLoss::getNegative(
    int32_t target,
    std::minstd_rand& rng) {
  int32_t negative;
  do {
    negative = negatives_[uniform_(rng)];
  } while (target == negative);
  return negative;
}
```


## N03 原生输入词 ID 与 EOS 特判


来源：[dictionary.cc](<C:/Projects/LLM/Gensim/references/sources/fastText-0.9.2/src/dictionary.cc:196>)，原文件第 196–205 行。


```cpp

void Dictionary::initNgrams() {
  for (size_t i = 0; i < size_; i++) {
    std::string word = BOW + words_[i].word + EOW;
    words_[i].subwords.clear();
    words_[i].subwords.push_back(i);
    if (words_[i].word != EOS) {
      computeSubwords(word, words_[i].subwords);
    }
  }
```


## N04 原生 SG／CBOW 的输入特征构建


来源：[fasttext.cc](<C:/Projects/LLM/Gensim/references/sources/fastText-0.9.2/src/fasttext.cc:387>)，原文件第 387–421 行。


```cpp
void FastText::cbow(
    Model::State& state,
    real lr,
    const std::vector<int32_t>& line) {
  std::vector<int32_t> bow;
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(state.rng);
    bow.clear();
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w + c]);
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    model_->update(bow, line, w, lr, state);
  }
}

void FastText::skipgram(
    Model::State& state,
    real lr,
    const std::vector<int32_t>& line) {
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(state.rng);
    const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w]);
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        model_->update(ngrams, line, w + c, lr, state);
      }
    }
  }
}

```


## N05 原生梯度归一化开关


来源：[fasttext.cc](<C:/Projects/LLM/Gensim/references/sources/fastText-0.9.2/src/fasttext.cc:762>)，原文件第 762–766 行。


```cpp
  }
  output_ = createTrainOutputMatrix();
  quant_ = false;
  auto loss = createLoss(output_);
  bool normalizeGradient = (args_->model == model_name::sup);
```


## 完整源码、授权与版本边界


Gensim 源码遵循 LGPL-2.1；原生 fastText 源码遵循 MIT。摘录保留原始注释；再分发源码应同时保留原许可证。本项目只审查 4.4.0／0.9.2，不把结论外推到其它版本。


Gensim 许可证：[LICENSE](<C:/Projects/LLM/Gensim/artifacts/reviews/fasttext_paper_engineering/sources/LICENSE>)；fastText 许可证：[LICENSE](<C:/Projects/LLM/Gensim/references/sources/fastText-0.9.2/LICENSE>)。


| 当前环境源码副本 | SHA-256 | 与项目已有副本比较 |
| --- | --- | --- |
| [fasttext.py](<C:/Projects/LLM/Gensim/artifacts/reviews/fasttext_paper_engineering/sources/fasttext.py>) | `a70e7c7b084ee8c9e03267023619d11d5e10a3dfe730e454ca95090c762f7770` | 文本一致（换行统一后） |
| [fasttext_inner.pyx](<C:/Projects/LLM/Gensim/artifacts/reviews/fasttext_paper_engineering/sources/fasttext_inner.pyx>) | `9c0d26f2a0409ff0ad08fa9598f05c941e200144929d5a7adabc4bb0460ecb62` | 文本一致（换行统一后） |
| [fasttext_inner.pxd](<C:/Projects/LLM/Gensim/artifacts/reviews/fasttext_paper_engineering/sources/fasttext_inner.pxd>) | `c005ec4dc47b7814442e0fdf667c64decdfa2561e6363b09bc3bced0d3a33fa7` | 本次新增存档 |
| [fasttext_corpusfile.pyx](<C:/Projects/LLM/Gensim/artifacts/reviews/fasttext_paper_engineering/sources/fasttext_corpusfile.pyx>) | `e8d3c8791dd7c4a3825433192074eea93431b43f3b303e92f5588acb6d395221` | 本次新增存档 |
| [word2vec.py](<C:/Projects/LLM/Gensim/artifacts/reviews/fasttext_paper_engineering/sources/word2vec.py>) | `d15cb29e9602719db96847008d91f47998a44ca5c575bc517de106bf60a711bf` | 文本一致（换行统一后） |
| [keyedvectors.py](<C:/Projects/LLM/Gensim/artifacts/reviews/fasttext_paper_engineering/sources/keyedvectors.py>) | `e1d05992c5d401a386f13bf29d61ed7be59af1b4a79d01b5e79cd29c8a06fa7e` | 文本一致（换行统一后） |


上述散列针对随 wheel 安装的源码。已单独记录实际加载 `.pyd` 的散列，并用数值探针验证本报告涉及的编译内核行为；没有声称只凭 `.pyx` 的散列能证明二进制构建完全相同。
