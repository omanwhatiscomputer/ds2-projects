package scalation.mathstat

import java.lang.foreign.*
import java.lang.foreign.ValueLayout.*
import scala.util.Try

object CudaTensorOps:

  private val libKernels = Try(SymbolLookup.libraryLookup("src/main/scala/scalation/mathstat/libC/libcudakernels.so", Arena.global())).toOption
  private val linker     = Linker.nativeLinker()

  private lazy val tensorOpAddr: MemorySegment =
    libKernels.get.find("gpuVectorOp").orElseThrow(() => new RuntimeException("Cannot find gpuVectorOp"))

  private lazy val tensorScalarOpAddr: MemorySegment =
    libKernels.get.find("gpuVectorScalarOp").orElseThrow(() => new RuntimeException("Cannot find gpuVectorScalarOp"))

  private lazy val tensorGlobalSumAddr: MemorySegment =
    libKernels.get.find("gpuMatrixGlobalSum").orElseThrow(() => new RuntimeException("Cannot find gpuMatrixGlobalSum"))

  private lazy val tensorContractionAddr: MemorySegment =
    libKernels.get.find("gpuTensorContraction").orElseThrow(() => new RuntimeException("Cannot find gpuTensorContraction"))

  // -------------------------------------------------------------------------
  // Flatten / unflatten helpers
  // -------------------------------------------------------------------------

  def flatten(t: TensorD): Array[Double] =
    val flat = new Array[Double](t.dim * t.dim2 * t.dim3)
    var i = 0
    while i < t.dim do
      var j = 0
      while j < t.dim2 do
        System.arraycopy(t.v(i)(j), 0, flat, i * t.dim2 * t.dim3 + j * t.dim3, t.dim3)
        j += 1
      i += 1
    flat

  def unflatten(flat: Array[Double], d1: Int, d2: Int, d3: Int): Array[Array[Array[Double]]] =
    val arr = Array.ofDim[Double](d1, d2, d3)
    var i = 0
    while i < d1 do
      var j = 0
      while j < d2 do
        System.arraycopy(flat, i * d2 * d3 + j * d3, arr(i)(j), 0, d3)
        j += 1
      i += 1
    arr

  // -------------------------------------------------------------------------
  // Element-wise tensor op (tensor op tensor)
  // -------------------------------------------------------------------------

  private def invokeTensorOp(a: Array[Double], b: Array[Double], n: Int, op: Int): Array[Double] =
    val kernelHandle = linker.downcallHandle(
      tensorOpAddr,
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT)
    )
    val result   = new Array[Double](n)
    val byteSize = n * 8L
    val arena    = Arena.ofConfined()
    try
      val segA   = arena.allocate(byteSize)
      val segB   = arena.allocate(byteSize)
      val segRes = arena.allocate(byteSize)
      MemorySegment.copy(a, 0, segA, ValueLayout.JAVA_DOUBLE, 0, n)
      MemorySegment.copy(b, 0, segB, ValueLayout.JAVA_DOUBLE, 0, n)
      kernelHandle.invoke(segA, segB, segRes, n, op)
      MemorySegment.copy(segRes, ValueLayout.JAVA_DOUBLE, 0, result, 0, n)
    finally
      arena.close()
    result

  def tensorOp(a: TensorD, b: TensorD, op: Int): Option[Array[Array[Array[Double]]]] =
    if !CudaVectorOps.isAvailable || a.dim != b.dim || a.dim2 != b.dim2 || a.dim3 != b.dim3 then return None
    val n = a.dim * a.dim2 * a.dim3
    Try(unflatten(invokeTensorOp(flatten(a), flatten(b), n, op), a.dim, a.dim2, a.dim3)).toOption

  // -------------------------------------------------------------------------
  // Element-wise scalar op (tensor op scalar)
  // -------------------------------------------------------------------------

  private def invokeTensorScalarOp(a: Array[Double], scalar: Double, n: Int, op: Int): Array[Double] =
    val kernelHandle = linker.downcallHandle(
      tensorScalarOpAddr,
      FunctionDescriptor.ofVoid(ADDRESS, JAVA_DOUBLE, ADDRESS, JAVA_INT, JAVA_INT)
    )
    val result   = new Array[Double](n)
    val byteSize = n * 8L
    val arena    = Arena.ofConfined()
    try
      val segA   = arena.allocate(byteSize)
      val segRes = arena.allocate(byteSize)
      MemorySegment.copy(a, 0, segA, ValueLayout.JAVA_DOUBLE, 0, n)
      kernelHandle.invoke(segA, scalar, segRes, n, op)
      MemorySegment.copy(segRes, ValueLayout.JAVA_DOUBLE, 0, result, 0, n)
    finally
      arena.close()
    result

  def tensorScalarOp(a: TensorD, scalar: Double, op: Int): Option[Array[Array[Array[Double]]]] =
    if !CudaVectorOps.isAvailable then return None
    val n = a.dim * a.dim2 * a.dim3
    Try(unflatten(invokeTensorScalarOp(flatten(a), scalar, n, op), a.dim, a.dim2, a.dim3)).toOption

  // -------------------------------------------------------------------------
  // Global sum reduction
  // -------------------------------------------------------------------------

  def globalSum(a: TensorD): Option[Double] =
    if !CudaVectorOps.isAvailable then return None
    val flat = flatten(a)
    val n    = flat.length
    Try {
      val kernelHandle = linker.downcallHandle(
        tensorGlobalSumAddr,
        FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_INT)
      )
      val arena = Arena.ofConfined()
      try
        val segA   = arena.allocate(n * 8L)
        val segRes = arena.allocate(8L)
        MemorySegment.copy(flat, 0, segA, ValueLayout.JAVA_DOUBLE, 0, n)
        kernelHandle.invoke(segA, segRes, n)
        segRes.get(ValueLayout.JAVA_DOUBLE, 0)
      finally
        arena.close()
    }.toOption

  // -------------------------------------------------------------------------
  // Tensor contraction: e(i,j,k) = sum_{l1,l2,l3} b(i,l1)*c(j,l2)*d(k,l3)*t(l1,l2,l3)
  // -------------------------------------------------------------------------

  def tensorContraction(t: TensorD, b: MatrixD, c: MatrixD, d: MatrixD): Option[Array[Array[Array[Double]]]] =
    if !CudaVectorOps.isAvailable then return None
    val L1 = t.dim;  val L2 = t.dim2; val L3 = t.dim3
    val M1 = b.dim;  val M2 = c.dim;  val M3 = d.dim
    def flatMat(m: MatrixD, rows: Int, cols: Int): Array[Double] =
      val flat = new Array[Double](rows * cols)
      var i = 0
      while i < rows do { System.arraycopy(m.v(i), 0, flat, i * cols, cols); i += 1 }
      flat
    val flatT = flatten(t)
    val flatB = flatMat(b, M1, L1)
    val flatC = flatMat(c, M2, L2)
    val flatD = flatMat(d, M3, L3)
    val flatE = new Array[Double](M1 * M2 * M3)
    Try {
      val kernelHandle = linker.downcallHandle(
        tensorContractionAddr,
        FunctionDescriptor.ofVoid(
          ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS,
          JAVA_INT, JAVA_INT, JAVA_INT,
          JAVA_INT, JAVA_INT, JAVA_INT
        )
      )
      val arena = Arena.ofConfined()
      try
        val segT = arena.allocate(flatT.length * 8L)
        val segB = arena.allocate(flatB.length * 8L)
        val segC = arena.allocate(flatC.length * 8L)
        val segD = arena.allocate(flatD.length * 8L)
        val segE = arena.allocate(flatE.length * 8L)
        MemorySegment.copy(flatT, 0, segT, ValueLayout.JAVA_DOUBLE, 0, flatT.length)
        MemorySegment.copy(flatB, 0, segB, ValueLayout.JAVA_DOUBLE, 0, flatB.length)
        MemorySegment.copy(flatC, 0, segC, ValueLayout.JAVA_DOUBLE, 0, flatC.length)
        MemorySegment.copy(flatD, 0, segD, ValueLayout.JAVA_DOUBLE, 0, flatD.length)
        kernelHandle.invoke(segT, segB, segC, segD, segE, L1, L2, L3, M1, M2, M3)
        MemorySegment.copy(segE, ValueLayout.JAVA_DOUBLE, 0, flatE, 0, flatE.length)
      finally
        arena.close()
      unflatten(flatE, M1, M2, M3)
    }.toOption

end CudaTensorOps
