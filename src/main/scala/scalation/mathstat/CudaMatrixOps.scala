package scalation.mathstat

import java.lang.foreign.*
import java.lang.foreign.ValueLayout.*
import scala.util.Try

object CudaMatrixOps:

  private val libKernels = Try(SymbolLookup.libraryLookup("src/main/scala/scalation/mathstat/libC/libcudakernels.so", Arena.global())).toOption
  private val linker     = Linker.nativeLinker()

  private val OP_ADD = 0
  private val OP_SUB = 1
  private val OP_MUL = 2
  private val OP_DIV = 3

//  private val opName = Map(OP_ADD -> "add", OP_SUB -> "sub", OP_MUL -> "mul", OP_DIV -> "div")

  // lazy: only evaluated on first dispatch call, which is guarded by CudaVectorOps.isAvailable
  private lazy val matrixOpAddr: MemorySegment =
    libKernels.get.find("gpuMatrixOp")
      .orElseThrow(() => new RuntimeException("Cannot find gpuMatrixOp"))

  private lazy val matrixScalarOpAddr: MemorySegment =
    libKernels.get.find("gpuMatrixScalarOp")
      .orElseThrow(() => new RuntimeException("Cannot find gpuMatrixScalarOp"))

  private lazy val matrixRowVecOpAddr: MemorySegment =
    libKernels.get.find("gpuMatrixRowVecOp")
      .orElseThrow(() => new RuntimeException("Cannot find gpuMatrixRowVecOp"))

  private lazy val matrixColVecOpAddr: MemorySegment =
    libKernels.get.find("gpuMatrixColVecOp")
      .orElseThrow(() => new RuntimeException("Cannot find gpuMatrixColVecOp"))

  private def flatten(m: Array[Array[Double]], rows: Int, cols: Int): Array[Double] =
    val flat = new Array[Double](rows * cols)
    var i = 0
    while i < rows do
      System.arraycopy(m(i), 0, flat, i * cols, cols)
      i += 1
    flat

  private def unflatten(flat: Array[Double], rows: Int, cols: Int): Array[Array[Double]] =
    val m = Array.ofDim[Double](rows, cols)
    var i = 0
    while i < rows do
      System.arraycopy(flat, i * cols, m(i), 0, cols)
      i += 1
    m

  private def invokeMatrixKernel(kernelAddr: MemorySegment, a: Array[Array[Double]], b: Array[Array[Double]], result: Array[Array[Double]], rows: Int, cols: Int, op: Int): Unit =
    val kernelHandle = linker.downcallHandle(
      kernelAddr,
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT)
    )
    val n        = rows * cols
    val byteSize = n * 8L
    val flatA    = flatten(a, rows, cols)
    val flatB    = flatten(b, rows, cols)
    val flatRes  = new Array[Double](n)
    val arena    = Arena.ofConfined()
    try
      val segA   = arena.allocate(byteSize)
      val segB   = arena.allocate(byteSize)
      val segRes = arena.allocate(byteSize)
      MemorySegment.copy(flatA, 0, segA, ValueLayout.JAVA_DOUBLE, 0, n)
      MemorySegment.copy(flatB, 0, segB, ValueLayout.JAVA_DOUBLE, 0, n)
      kernelHandle.invoke(segA, segB, segRes, rows, cols, op)
      MemorySegment.copy(segRes, ValueLayout.JAVA_DOUBLE, 0, flatRes, 0, n)
    finally
      arena.close()
    end try
    val m = unflatten(flatRes, rows, cols)
    var i = 0
    while i < rows do
      System.arraycopy(m(i), 0, result(i), 0, cols)
      i += 1
  end invokeMatrixKernel

  private def invokeMatrixScalarKernel(kernelAddr: MemorySegment, a: Array[Array[Double]], scalar: Double, result: Array[Array[Double]], rows: Int, cols: Int, op: Int): Unit =
    val kernelHandle = linker.downcallHandle(
      kernelAddr,
      FunctionDescriptor.ofVoid(ADDRESS, JAVA_DOUBLE, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT)
    )
    val n        = rows * cols
    val byteSize = n * 8L
    val flatA    = flatten(a, rows, cols)
    val flatRes  = new Array[Double](n)
    val arena    = Arena.ofConfined()
    try
      val segA   = arena.allocate(byteSize)
      val segRes = arena.allocate(byteSize)
      MemorySegment.copy(flatA, 0, segA, ValueLayout.JAVA_DOUBLE, 0, n)
      kernelHandle.invoke(segA, scalar, segRes, rows, cols, op)
      MemorySegment.copy(segRes, ValueLayout.JAVA_DOUBLE, 0, flatRes, 0, n)
    finally
      arena.close()
    end try
    val m = unflatten(flatRes, rows, cols)
    var i = 0
    while i < rows do
      System.arraycopy(m(i), 0, result(i), 0, cols)
      i += 1
  end invokeMatrixScalarKernel

  private def dispatch(a: Array[Array[Double]], b: Array[Array[Double]], rows: Int, cols: Int, op: Int): Option[Array[Array[Double]]] =
    if !CudaVectorOps.isAvailable then return None
    val result = Array.ofDim[Double](rows, cols)
    try
      invokeMatrixKernel(matrixOpAddr, a, b, result, rows, cols, op)
//      println(s"[CudaMatrixOps] GPU executed: matrix ${opName(op)} matrix on ${rows}x${cols} elements")
      Some(result)
    catch case e: Throwable =>
      e.printStackTrace()
      None

  private def dispatchScalar(a: Array[Array[Double]], scalar: Double, rows: Int, cols: Int, op: Int): Option[Array[Array[Double]]] =
    if !CudaVectorOps.isAvailable then return None
    val result = Array.ofDim[Double](rows, cols)
    try
      invokeMatrixScalarKernel(matrixScalarOpAddr, a, scalar, result, rows, cols, op)
//      println(s"[CudaMatrixOps] GPU executed: matrix ${opName(op)} scalar on ${rows}x${cols} elements")
      Some(result)
    catch case e: Throwable =>
      e.printStackTrace()
      None

  private def invokeMatrixVecKernel(kernelAddr: MemorySegment, a: Array[Array[Double]], vec: Array[Double], result: Array[Array[Double]], rows: Int, cols: Int, op: Int): Unit =
    val kernelHandle = linker.downcallHandle(
      kernelAddr,
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT)
    )
    val n        = rows * cols
    val byteSize = n * 8L
    val vecSize  = vec.length * 8L
    val flatA    = flatten(a, rows, cols)
    val flatRes  = new Array[Double](n)
    val arena    = Arena.ofConfined()
    try
      val segA   = arena.allocate(byteSize)
      val segVec = arena.allocate(vecSize)
      val segRes = arena.allocate(byteSize)
      MemorySegment.copy(flatA, 0, segA,   ValueLayout.JAVA_DOUBLE, 0, n)
      MemorySegment.copy(vec,   0, segVec, ValueLayout.JAVA_DOUBLE, 0, vec.length)
      kernelHandle.invoke(segA, segVec, segRes, rows, cols, op)
      MemorySegment.copy(segRes, ValueLayout.JAVA_DOUBLE, 0, flatRes, 0, n)
    finally
      arena.close()
    end try
    val m = unflatten(flatRes, rows, cols)
    var i = 0
    while i < rows do
      System.arraycopy(m(i), 0, result(i), 0, cols)
      i += 1
  end invokeMatrixVecKernel



  private def dispatchRowVec(a: Array[Array[Double]], vec: Array[Double], rows: Int, cols: Int, op: Int): Option[Array[Array[Double]]] =

    if !CudaVectorOps.isAvailable then
      return None

    val result = Array.ofDim[Double](rows, cols)
    try
      invokeMatrixVecKernel(matrixRowVecOpAddr, a, vec, result, rows, cols, op)
//      println(s"[CudaMatrixOps] GPU executed: matrix ${opName(op)} rowVec on ${rows}x${cols} elements")
      Some(result)
    catch case e: Throwable =>
      println(s"EXCEPTION: ${e.getClass.getName}: ${e.getMessage}")
      e.printStackTrace()
      None



  private def dispatchColVec(a: Array[Array[Double]], vec: Array[Double], rows: Int, cols: Int, op: Int): Option[Array[Array[Double]]] =

    if !CudaVectorOps.isAvailable then
      return None

    val result = Array.ofDim[Double](rows, cols)
    try
      invokeMatrixVecKernel(matrixColVecOpAddr, a, vec, result, rows, cols, op)
//      println(s"[CudaMatrixOps] GPU executed: matrix ${opName(op)} colVec on ${rows}x${cols} elements")
      Some(result)
    catch case e: Throwable =>
      println(s"EXCEPTION: ${e.getClass.getName}: ${e.getMessage}")
      e.printStackTrace()
      None

  // Public API (matrix op row vector, broadcast)
  def addRowVec(a: Array[Array[Double]], vec: Array[Double], rows: Int, cols: Int): Option[Array[Array[Double]]] = dispatchRowVec(a, vec, rows, cols, OP_ADD)
  def subRowVec(a: Array[Array[Double]], vec: Array[Double], rows: Int, cols: Int): Option[Array[Array[Double]]] = dispatchRowVec(a, vec, rows, cols, OP_SUB)
  def mulRowVec(a: Array[Array[Double]], vec: Array[Double], rows: Int, cols: Int): Option[Array[Array[Double]]] = dispatchRowVec(a, vec, rows, cols, OP_MUL)
  def divRowVec(a: Array[Array[Double]], vec: Array[Double], rows: Int, cols: Int): Option[Array[Array[Double]]] = dispatchRowVec(a, vec, rows, cols, OP_DIV)

  // Public API (matrix op col vector, broadcast)
  def addColVec(a: Array[Array[Double]], vec: Array[Double], rows: Int, cols: Int): Option[Array[Array[Double]]] = dispatchColVec(a, vec, rows, cols, OP_ADD)
  def subColVec(a: Array[Array[Double]], vec: Array[Double], rows: Int, cols: Int): Option[Array[Array[Double]]] = dispatchColVec(a, vec, rows, cols, OP_SUB)
  def mulColVec(a: Array[Array[Double]], vec: Array[Double], rows: Int, cols: Int): Option[Array[Array[Double]]] = dispatchColVec(a, vec, rows, cols, OP_MUL)

  // Public API (matrix op matrix - element-wise)
  def add(a: Array[Array[Double]], b: Array[Array[Double]], rows: Int, cols: Int): Option[Array[Array[Double]]] = dispatch(a, b, rows, cols, OP_ADD)
  def sub(a: Array[Array[Double]], b: Array[Array[Double]], rows: Int, cols: Int): Option[Array[Array[Double]]] = dispatch(a, b, rows, cols, OP_SUB)
  def mul(a: Array[Array[Double]], b: Array[Array[Double]], rows: Int, cols: Int): Option[Array[Array[Double]]] = dispatch(a, b, rows, cols, OP_MUL)
  def div(a: Array[Array[Double]], b: Array[Array[Double]], rows: Int, cols: Int): Option[Array[Array[Double]]] = dispatch(a, b, rows, cols, OP_DIV)

  // Public API (matrix op scalar)
  def addScalar(a: Array[Array[Double]], s: Double, rows: Int, cols: Int): Option[Array[Array[Double]]] = dispatchScalar(a, s, rows, cols, OP_ADD)
  def subScalar(a: Array[Array[Double]], s: Double, rows: Int, cols: Int): Option[Array[Array[Double]]] = dispatchScalar(a, s, rows, cols, OP_SUB)
  def mulScalar(a: Array[Array[Double]], s: Double, rows: Int, cols: Int): Option[Array[Array[Double]]] = dispatchScalar(a, s, rows, cols, OP_MUL)
  def divScalar(a: Array[Array[Double]], s: Double, rows: Int, cols: Int): Option[Array[Array[Double]]] = dispatchScalar(a, s, rows, cols, OP_DIV)

  // ── New symbol addresses ────────────────────────────────────────────────

  private lazy val matrixMulAddr: MemorySegment =
    libKernels.get.find("gpuMatrixMul").orElseThrow(() => new RuntimeException("Cannot find gpuMatrixMul"))

  private lazy val matrixVecMulAddr: MemorySegment =
    libKernels.get.find("gpuMatrixVecMul").orElseThrow(() => new RuntimeException("Cannot find gpuMatrixVecMul"))

  private lazy val matrixTransVecMulAddr: MemorySegment =
    libKernels.get.find("gpuMatrixTransVecMul").orElseThrow(() => new RuntimeException("Cannot find gpuMatrixTransVecMul"))

  private lazy val matrixColSumAddr: MemorySegment =
    libKernels.get.find("gpuMatrixColSum").orElseThrow(() => new RuntimeException("Cannot find gpuMatrixColSum"))

  private lazy val matrixRowSumAddr: MemorySegment =
    libKernels.get.find("gpuMatrixRowSum").orElseThrow(() => new RuntimeException("Cannot find gpuMatrixRowSum"))

  private lazy val matrixGlobalSumAddr: MemorySegment =
    libKernels.get.find("gpuMatrixGlobalSum").orElseThrow(() => new RuntimeException("Cannot find gpuMatrixGlobalSum"))

  private lazy val matrixGlobalMinAddr: MemorySegment =
    libKernels.get.find("gpuMatrixGlobalMin").orElseThrow(() => new RuntimeException("Cannot find gpuMatrixGlobalMin"))

  private lazy val matrixGlobalMaxAddr: MemorySegment =
    libKernels.get.find("gpuMatrixGlobalMax").orElseThrow(() => new RuntimeException("Cannot find gpuMatrixGlobalMax"))

  private lazy val matrixColMinAddr: MemorySegment =
    libKernels.get.find("gpuMatrixColMin").orElseThrow(() => new RuntimeException("Cannot find gpuMatrixColMin"))

  private lazy val matrixColMaxAddr: MemorySegment =
    libKernels.get.find("gpuMatrixColMax").orElseThrow(() => new RuntimeException("Cannot find gpuMatrixColMax"))

  // ── GEMM invoke: (segA, segB, segC, m, k, n) ───────────────────────────

  private def invokeGemm(kernelAddr: MemorySegment, a: Array[Array[Double]], b: Array[Array[Double]], result: Array[Array[Double]], m: Int, k: Int, n: Int): Unit =
    val kernelHandle = linker.downcallHandle(
      kernelAddr,
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT)
    )
    val flatA   = flatten(a, m, k)
    val flatB   = flatten(b, k, n)
    val flatRes = new Array[Double](m * n)
    val arena   = Arena.ofConfined()
    try
      val segA   = arena.allocate(m.toLong * k * 8)
      val segB   = arena.allocate(k.toLong * n * 8)
      val segRes = arena.allocate(m.toLong * n * 8)
      MemorySegment.copy(flatA, 0, segA,   ValueLayout.JAVA_DOUBLE, 0, m * k)
      MemorySegment.copy(flatB, 0, segB,   ValueLayout.JAVA_DOUBLE, 0, k * n)
      kernelHandle.invoke(segA, segB, segRes, m, k, n)
      MemorySegment.copy(segRes, ValueLayout.JAVA_DOUBLE, 0, flatRes, 0, m * n)
    finally
      arena.close()
    end try
    val mat = unflatten(flatRes, m, n)
    var i = 0
    while i < m do
      System.arraycopy(mat(i), 0, result(i), 0, n)
      i += 1
  end invokeGemm

  // ── GEMV invoke: (segA, segVec, segResult, dim1, dim2) ─────────────────
  // Used for both GEMV (dim1=rows, dim2=cols, resultLen=rows)
  // and transposed GEMV (dim1=rows, dim2=cols, resultLen=cols).

  private def invokeGemvKernel(kernelAddr: MemorySegment, a: Array[Array[Double]], vec: Array[Double], resultVec: Array[Double], rows: Int, cols: Int): Unit =
    val kernelHandle = linker.downcallHandle(
      kernelAddr,
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT)
    )
    val flatA = flatten(a, rows, cols)
    val arena = Arena.ofConfined()
    try
      val segA   = arena.allocate(rows.toLong * cols * 8)
      val segVec = arena.allocate(vec.length.toLong * 8)
      val segRes = arena.allocate(resultVec.length.toLong * 8)
      MemorySegment.copy(flatA, 0, segA,   ValueLayout.JAVA_DOUBLE, 0, rows * cols)
      MemorySegment.copy(vec,   0, segVec, ValueLayout.JAVA_DOUBLE, 0, vec.length)
      kernelHandle.invoke(segA, segVec, segRes, rows, cols)
      MemorySegment.copy(segRes, ValueLayout.JAVA_DOUBLE, 0, resultVec, 0, resultVec.length)
    finally
      arena.close()
    end try
  end invokeGemvKernel

  // ── Global scalar reduction invoke: (segFlat, segResult, n) ────────────

  private def invokeGlobalReduction(kernelAddr: MemorySegment, flat: Array[Double]): Double =
    val kernelHandle = linker.downcallHandle(
      kernelAddr,
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_INT)
    )
    val arena = Arena.ofConfined()
    try
      val segA   = arena.allocate(flat.length.toLong * 8)
      val segRes = arena.allocate(8L)
      MemorySegment.copy(flat, 0, segA, ValueLayout.JAVA_DOUBLE, 0, flat.length)
      kernelHandle.invoke(segA, segRes, flat.length)
      segRes.get(ValueLayout.JAVA_DOUBLE, 0)
    finally
      arena.close()
    end try
  end invokeGlobalReduction

  // ── Col/row vector reduction invoke: (segFlat, segResult, rows, cols) ───

  private def invokeVectorReduction(kernelAddr: MemorySegment, flat: Array[Double], rows: Int, cols: Int, resultLen: Int): Array[Double] =
    val kernelHandle = linker.downcallHandle(
      kernelAddr,
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT)
    )
    val resultArr = new Array[Double](resultLen)
    val arena     = Arena.ofConfined()
    try
      val segA   = arena.allocate(flat.length.toLong * 8)
      val segRes = arena.allocate(resultLen.toLong * 8)
      MemorySegment.copy(flat, 0, segA, ValueLayout.JAVA_DOUBLE, 0, flat.length)
      kernelHandle.invoke(segA, segRes, rows, cols)
      MemorySegment.copy(segRes, ValueLayout.JAVA_DOUBLE, 0, resultArr, 0, resultLen)
    finally
      arena.close()
    end try
    resultArr
  end invokeVectorReduction

  // ── Dispatch functions ──────────────────────────────────────────────────

  private def dispatchGemm(a: Array[Array[Double]], b: Array[Array[Double]], m: Int, k: Int, n: Int): Option[Array[Array[Double]]] =
    if !CudaVectorOps.isAvailable then return None
    val result = Array.ofDim[Double](m, n)
    try
      invokeGemm(matrixMulAddr, a, b, result, m, k, n)
      Some(result)
    catch case e: Throwable =>
      e.printStackTrace()
      None

  private def dispatchGemv(a: Array[Array[Double]], vec: Array[Double], rows: Int, cols: Int): Option[Array[Double]] =
    if !CudaVectorOps.isAvailable then return None
    val result = new Array[Double](rows)
    try
      invokeGemvKernel(matrixVecMulAddr, a, vec, result, rows, cols)
      Some(result)
    catch case e: Throwable =>
      e.printStackTrace()
      None

  private def dispatchTransGemv(a: Array[Array[Double]], vec: Array[Double], rows: Int, cols: Int): Option[Array[Double]] =
    if !CudaVectorOps.isAvailable then return None
    val result = new Array[Double](cols)
    try
      invokeGemvKernel(matrixTransVecMulAddr, a, vec, result, rows, cols)
      Some(result)
    catch case e: Throwable =>
      e.printStackTrace()
      None

  private def dispatchGlobalReduction(kernelAddr: MemorySegment, a: Array[Array[Double]], rows: Int, cols: Int): Option[Double] =
    if !CudaVectorOps.isAvailable then return None
    try
      Some(invokeGlobalReduction(kernelAddr, flatten(a, rows, cols)))
    catch case e: Throwable =>
      e.printStackTrace()
      None

  private def dispatchVectorReduction(kernelAddr: MemorySegment, a: Array[Array[Double]], rows: Int, cols: Int, resultLen: Int): Option[Array[Double]] =
    if !CudaVectorOps.isAvailable then return None
    try
      Some(invokeVectorReduction(kernelAddr, flatten(a, rows, cols), rows, cols, resultLen))
    catch case e: Throwable =>
      e.printStackTrace()
      None

  // ── Public API (GEMM, GEMV, reductions) ────────────────────────────────

  def matrixMul(a: Array[Array[Double]], b: Array[Array[Double]], m: Int, k: Int, n: Int): Option[Array[Array[Double]]] =
    dispatchGemm(a, b, m, k, n)

  def matrixVecMul(a: Array[Array[Double]], vec: Array[Double], rows: Int, cols: Int): Option[Array[Double]] =
    dispatchGemv(a, vec, rows, cols)

  def matrixTransVecMul(a: Array[Array[Double]], vec: Array[Double], rows: Int, cols: Int): Option[Array[Double]] =
    dispatchTransGemv(a, vec, rows, cols)

  def globalSum(a: Array[Array[Double]], rows: Int, cols: Int): Option[Double] =
    dispatchGlobalReduction(matrixGlobalSumAddr, a, rows, cols)

  def globalMin(a: Array[Array[Double]], rows: Int, cols: Int): Option[Double] =
    dispatchGlobalReduction(matrixGlobalMinAddr, a, rows, cols)

  def globalMax(a: Array[Array[Double]], rows: Int, cols: Int): Option[Double] =
    dispatchGlobalReduction(matrixGlobalMaxAddr, a, rows, cols)

  def colSum(a: Array[Array[Double]], rows: Int, cols: Int): Option[Array[Double]] =
    dispatchVectorReduction(matrixColSumAddr, a, rows, cols, cols)

  def rowSum(a: Array[Array[Double]], rows: Int, cols: Int): Option[Array[Double]] =
    dispatchVectorReduction(matrixRowSumAddr, a, rows, cols, rows)

  def colMin(a: Array[Array[Double]], rows: Int, cols: Int): Option[Array[Double]] =
    dispatchVectorReduction(matrixColMinAddr, a, rows, cols, cols)

  def colMax(a: Array[Array[Double]], rows: Int, cols: Int): Option[Array[Double]] =
    dispatchVectorReduction(matrixColMaxAddr, a, rows, cols, cols)

end CudaMatrixOps
