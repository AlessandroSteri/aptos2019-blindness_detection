
hen_kappa(labels,
                predictions_idx,
                num_classes,
                weights=None,
                metrics_collections=None,
                updates_collections=None,
                name=None):
  """Calculates Cohen's kappa.
  [Cohen's kappa](https://en.wikipedia.org/wiki/Cohen's_kappa) is a statistic
  that measures inter-annotator agreement.
  The `cohen_kappa` function calculates the confusion matrix, and creates three
  local variables to compute the Cohen's kappa: `po`, `pe_row`, and `pe_col`,
  which refer to the diagonal part, rows and columns totals of the confusion
  matrix, respectively. This value is ultimately returned as `kappa`, an
  idempotent operation that is calculated by
      pe = (pe_row * pe_col) / N
      k = (sum(po) - sum(pe)) / (N - sum(pe))
  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `kappa`. `update_op` weights each prediction by the corresponding value in
  `weights`.
  Class labels are expected to start at 0. E.g., if `num_classes`
  was three, then the possible labels would be [0, 1, 2].
  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.
  NOTE: Equivalent to `sklearn.metrics.cohen_kappa_score`, but the method
  doesn't support weighted matrix yet.
  Args:
    labels: 1-D `Tensor` of real labels for the classification task. Must be
      one of the following types: int16, int32, int64.
    predictions_idx: 1-D `Tensor` of predicted class indices for a given
      classification. Must have the same type as `labels`.
    num_classes: The possible number of labels.
    weights: Optional `Tensor` whose shape matches `predictions`.
    metrics_collections: An optional list of collections that `kappa` should be
      added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.
  Returns:
    kappa: Scalar float `Tensor` representing the current Cohen's kappa.
    update_op: `Operation` that increments `po`, `pe_row` and `pe_col`
      variables appropriately and whose value matches `kappa`.
  Raises:
    ValueError: If `num_classes` is less than 2, or `predictions` and `labels`
      have mismatched shapes, or if `weights` is not `None` and its shape
      doesn't match `predictions`, or if either `metrics_collections` or
      `updates_collections` are not a list or tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.contrib.metrics.cohen_kappa is not supported '
                       'when eager execution is enabled.')
  if num_classes < 2:
    raise ValueError('`num_classes` must be >= 2.'
                     'Found: {}'.format(num_classes))
  with variable_scope.variable_scope(name, 'cohen_kappa',
                                     (labels, predictions_idx, weights)):
    # Convert 2-dim (num, 1) to 1-dim (num,)
    labels.get_shape().with_rank_at_most(2)
    if labels.get_shape().ndims == 2:
      labels = array_ops.squeeze(labels, axis=[-1])
    predictions_idx, labels, weights = (
        metrics_impl._remove_squeezable_dimensions(  # pylint: disable=protected-access
            predictions=predictions_idx,
            labels=labels,
            weights=weights))
    predictions_idx.get_shape().assert_is_compatible_with(labels.get_shape())

    stat_dtype = (
        dtypes.int64
        if weights is None or weights.dtype.is_integer else dtypes.float32)
    po = metrics_impl.metric_variable((num_classes,), stat_dtype, name='po')
    pe_row = metrics_impl.metric_variable((num_classes,),
                                          stat_dtype,
                                          name='pe_row')
    pe_col = metrics_impl.metric_variable((num_classes,),
                                          stat_dtype,
                                          name='pe_col')

    # Table of the counts of agreement:
    counts_in_table = confusion_matrix.confusion_matrix(
        labels,
        predictions_idx,
        num_classes=num_classes,
        weights=weights,
        dtype=stat_dtype,
        name='counts_in_table')

    po_t = array_ops.diag_part(counts_in_table)
    pe_row_t = math_ops.reduce_sum(counts_in_table, axis=0)
    pe_col_t = math_ops.reduce_sum(counts_in_table, axis=1)
    update_po = state_ops.assign_add(po, po_t)
    update_pe_row = state_ops.assign_add(pe_row, pe_row_t)
    update_pe_col = state_ops.assign_add(pe_col, pe_col_t)

    def _calculate_k(po, pe_row, pe_col, name):
      po_sum = math_ops.reduce_sum(po)
      total = math_ops.reduce_sum(pe_row)
      pe_sum = math_ops.reduce_sum(
          math_ops.div_no_nan(
              math_ops.cast(pe_row * pe_col, dtypes.float64),
              math_ops.cast(total, dtypes.float64)))
      po_sum, pe_sum, total = (math_ops.cast(po_sum, dtypes.float64),
                               math_ops.cast(pe_sum, dtypes.float64),
                               math_ops.cast(total, dtypes.float64))
      # kappa = (po - pe) / (N - pe)
      k = metrics_impl._safe_scalar_div(  # pylint: disable=protected-access
          po_sum - pe_sum,
          total - pe_sum,
          name=name)
      return k

    kappa = _calculate_k(po, pe_row, pe_col, name='value')
    update_op = _calculate_k(
        update_po, update_pe_row, update_pe_col, name='update_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, kappa)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return kappa, update_op
