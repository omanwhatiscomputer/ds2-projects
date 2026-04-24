package scalation.mathstat

import java.lang.foreign.*
import java.lang.foreign.ValueLayout.*

object CudaMatrixOps:

  private val libKernels = SymbolLookup.libraryLookup("src/main/scala/scalation/mathstat/libC/libcudakernels.so", Arena.global())
  private val linker     = Linker.nativeLinker()

  private val OP_ADD = 0
  private val OP_SUB = 1
  private val OP_MUL = 2
  private val OP_DIV = 3

  private val opName = Map(OP_ADD -> "add", OP_SUB -> "sub", OP_MUL -> "mul", OP_DIV -> "div")

  private val matrixOpAddr: MemorySegment =
    libKernels.find("gpuMatrixOp")
      .orElseThrow(() => new RuntimeException("Cannot find gpuMatrixOp"))

  private val matrixScalarOpAddr: MemorySegment =
    libKernels.find("gpuMatrixScalarOp")
      .orElseThrow(() => new RuntimeException("Cannot find gpuMatrixScalarOp"))

  private val matrixRowVecOpAddr: MemorySegment =
    libKernels.find("gpuMatrixRowVecOp")
      .orElseThrow(() => new RuntimeException("Cannot find gpuMatrixRowVecOp"))

  private val matrixColVecOpAddr: MemorySegment =
    libKernels.find("gpuMatrixColVecOp")
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

end CudaMatrixOps
