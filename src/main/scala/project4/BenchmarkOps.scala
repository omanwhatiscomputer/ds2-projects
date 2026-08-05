package project4

import scalation.{time, banner}
import scalation.mathstat.{VectorD, MatrixD}

@main def BenchmarkOps(): Unit =

  val ROWS       = 3_000
  val COLS       = 3_000
  val iterations = 10

  val m   = MatrixD((0 until ROWS).map(_ => VectorD(Array.fill(COLS)(1.5))))
  val u   = VectorD(Array.fill(ROWS)(2.0))
  val y   = VectorD(Array.fill(ROWS)(1.0))

  // -------------------------------------------------------------------------
  // Old (sequential) implementations — logic from pre-commit diff
  // Uses only public API: m(i,j) for reads, m('?',j) for column extraction
  // -------------------------------------------------------------------------

  def old_prependCol(m: MatrixD, u: VectorD): MatrixD =
    MatrixD(IndexedSeq.tabulate(m.dim) { i =>
      VectorD(Array.tabulate(m.dim2 + 1)(j => if j == 0 then u(i) else m(i, j - 1)))
    })

  def old_crossAll(m: MatrixD): MatrixD =
    val cols = for i <- 0 until m.dim2; j <- 0 until i yield
      VectorD(Array.tabulate(m.dim)(r => m(r, i) * m(r, j)))
    MatrixD(cols.toIndexedSeq).transpose

  def old_crossAll3(m: MatrixD): MatrixD =
    val cols = for i <- 0 until m.dim2; j <- 0 until i; k <- 0 until j yield
      VectorD(Array.tabulate(m.dim)(r => m(r, i) * m(r, j) * m(r, k)))
    MatrixD(cols.toIndexedSeq).transpose

  def old_dot(m: MatrixD, y: VectorD): VectorD =
    VectorD(Array.tabulate(m.dim2) { j =>
      val col = m('?', j)
      var sum = 0.0
      var i   = 0
      while i < m.dim do { sum += col(i) * y(i); i += 1 }
      sum
    })

  def old_mmap_(m: MatrixD, f: VectorD => VectorD): MatrixD =
    MatrixD(m.indices2.map { j => f(m('?', j)) }).transpose

  def old_corr(m: MatrixD, y: VectorD, skip: Int = 0): VectorD =
    VectorD((skip until m.dim2).map { j => m('?', j).corr(y) })

  // -------------------------------------------------------------------------
  val squash: VectorD => VectorD = v => v.map(x => 1.0 / (1.0 + math.exp(-x)))

  // =========================================================================
  // +^: Prepend Column Vector
  // =========================================================================

  for i <- 0 until iterations do
    if i == iterations - 1 then
      banner(s"[+^:] Prepend Column Vector ($ROWS x $COLS)")
      println("[OLD sequential]"); time { old_prependCol(m, u) }
      println("[NEW parallel  ]"); time { u +^: m }

  // =========================================================================
  // crossAll — All 2-way interaction terms
  // =========================================================================

  val COLS_CROSS = 200
  val mc = MatrixD((0 until ROWS).map(_ => VectorD(Array.fill(COLS_CROSS)(1.5))))

  for i <- 0 until iterations do
    if i == iterations - 1 then
      banner(s"[crossAll] All 2-way terms ($ROWS x $COLS_CROSS -> $ROWS x ${COLS_CROSS*(COLS_CROSS-1)/2})")
      println("[OLD sequential]"); time { old_crossAll(mc) }
      println("[NEW parallel  ]"); time { mc.crossAll }

  // =========================================================================
  // crossAll3 — All 3-way interaction terms
  // =========================================================================

  val COLS_CROSS3 = 50
  val mc3 = MatrixD((0 until ROWS).map(_ => VectorD(Array.fill(COLS_CROSS3)(1.5))))

  for i <- 0 until iterations do
    if i == iterations - 1 then
      banner(s"[crossAll3] All 3-way terms ($ROWS x $COLS_CROSS3 -> $ROWS x ${COLS_CROSS3*(COLS_CROSS3-1)*(COLS_CROSS3-2)/6})")
      println("[OLD sequential]"); time { old_crossAll3(mc3) }
      println("[NEW parallel  ]"); time { mc3.crossAll3 }

  // =========================================================================
  // dot(VectorD) — column-wise projection onto y
  // =========================================================================

  for i <- 0 until iterations do
    if i == iterations - 1 then
      banner(s"[dot(VectorD)] Matrix-Vector Column Projection ($ROWS x $COLS)")
      println("[OLD sequential]"); time { old_dot(m, y) }
      println("[NEW parallel  ]"); time { m dot y }

  // =========================================================================
  // mmap_ — apply function to each column
  // =========================================================================

  for i <- 0 until iterations do
    if i == iterations - 1 then
      banner(s"[mmap_] Apply Function to Each Column ($ROWS x $COLS)")
      println("[OLD sequential]"); time { old_mmap_(m, squash) }
      println("[NEW parallel  ]"); time { m.mmap_(squash) }

  // =========================================================================
  // corr(VectorD) — correlation of each column with y
  // =========================================================================

  for i <- 0 until iterations do
    if i == iterations - 1 then
      banner(s"[corr(VectorD)] Column-Target Correlations ($ROWS x $COLS)")
      println("[OLD sequential]"); time { old_corr(m, y) }
      println("[NEW parallel  ]"); time { m.corr(y) }

end BenchmarkOps
