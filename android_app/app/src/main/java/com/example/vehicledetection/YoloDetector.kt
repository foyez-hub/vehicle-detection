package com.example.vehicledetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.util.Collections

class YoloDetector(private val context: Context) {

    private val stepSize = 640
    private val modelPath = "vehicle_detector.onnx"
    private var env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession

    private val labels = mapOf(
        0 to "Auto Rickshaw",
        1 to "Cycle Rickshaw",
        2 to "CNG/Tempo",
        3 to "Bus",
        4 to "Jeep/SUV",
        5 to "Microbus",
        6 to "Minibus",
        7 to "Motorcycle",
        8 to "Truck",
        9 to "Private Sedan Car",
        10 to "Trailer"
    )

    data class Detection(
        val box: FloatArray, // [x1, y1, x2, y2]
        val score: Float,
        val classId: Int
    )

    init {
        val modelBytes = context.assets.open(modelPath).readBytes()
        val options = OrtSession.SessionOptions()
        session = env.createSession(modelBytes, options)
    }

    fun detect(bitmap: Bitmap): Result {
        // 1. Preprocess
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, stepSize, stepSize, true)
        val floatBuffer = allocateFloatBuffer(resizedBitmap)
        val inputName = session.inputNames.iterator().next()
        val shape = longArrayOf(1, 3, stepSize.toLong(), stepSize.toLong())
        val inputTensor = OnnxTensor.createTensor(env, floatBuffer, shape)

        // 2. Inference
        val results = session.run(Collections.singletonMap(inputName, inputTensor))
        val output = results[0].value as Array<Array<FloatArray>> // [1, 15, 8400] for YOLOv8-n end2end=True? No, standard is [1, 15, 8400] where 15 = 4 (box) + 11 (classes)
        
        // 3. Postprocess (Standard YOLOv8 output: 1x(4+classes)x8400)
        // Note: The metadata suggested end2end=True, but usually that means 1x300x(4+1) or something.
        // Let's assume standard shape from metadata: names length is 11, so 4+11=15.
        // The test_onnx.py showed resize to 640x640.
        
        val detections = processOutput(output[0])
        val filteredDetections = applyNMS(detections)

        // 4. Draw & Count
        val resultBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(resultBitmap)
        val paint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 5f
            textSize = 40f
        }
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 40f
            backgroundColor = Color.RED
        }

        val counts = mutableMapOf<String, Int>()
        val scaleX = bitmap.width.toFloat() / stepSize
        val scaleY = bitmap.height.toFloat() / stepSize

        for (det in filteredDetections) {
            val label = labels[det.classId] ?: "Unknown"
            counts[label] = counts.getOrDefault(label, 0) + 1

            val x1 = det.box[0] * scaleX
            val y1 = det.box[1] * scaleY
            val x2 = det.box[2] * scaleX
            val y2 = det.box[3] * scaleY

            canvas.drawRect(x1, y1, x2, y2, paint)
            canvas.drawText("$label ${(det.score * 100).toInt()}%", x1, y1 - 10f, textPaint)
        }

        return Result(resultBitmap, filteredDetections.size, counts)
    }

    private fun allocateFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val buffer = FloatBuffer.allocate(3 * stepSize * stepSize)
        val pixels = IntArray(stepSize * stepSize)
        bitmap.getPixels(pixels, 0, stepSize, 0, 0, stepSize, stepSize)

        // RGB normalization
        for (i in 0 until stepSize * stepSize) {
            val pixel = pixels[i]
            buffer.put(i, ((pixel shr 16) and 0xFF) / 255.0f)
            buffer.put(i + stepSize * stepSize, ((pixel shr 8) and 0xFF) / 255.0f)
            buffer.put(i + 2 * stepSize * stepSize, (pixel and 0xFF) / 255.0f)
        }
        return buffer
    }

    private fun processOutput(output: Array<FloatArray>): List<Detection> {
        // Output shape is [15, 8400]
        val detections = mutableListOf<Detection>()
        val rows = output[0].size // 8400
        val cols = output.size    // 15

        for (i in 0 until rows) {
            var maxConf = 0f
            var classId = -1
            for (j in 4 until cols) {
                if (output[j][i] > maxConf) {
                    maxConf = output[j][i]
                    classId = j - 4
                }
            }

            if (maxConf > 0.45f) {
                val cx = output[0][i]
                val cy = output[1][i]
                val w = output[2][i]
                val h = output[3][i]
                val x1 = cx - w / 2
                val y1 = cy - h / 2
                val x2 = cx + w / 2
                val y2 = cy + h / 2
                detections.add(Detection(floatArrayOf(x1, y1, x2, y2), maxConf, classId))
            }
        }
        return detections
    }

    private fun applyNMS(detections: List<Detection>): List<Detection> {
        val res = mutableListOf<Detection>()
        val sortedDetections = detections.sortedByDescending { it.score }
        val visited = BooleanArray(sortedDetections.size)

        for (i in sortedDetections.indices) {
            if (visited[i]) continue
            res.add(sortedDetections[i])
            for (j in i + 1 until sortedDetections.size) {
                if (visited[j]) continue
                if (calculateIoU(sortedDetections[i].box, sortedDetections[j].box) > 0.45f) {
                    visited[j] = true
                }
            }
        }
        return res
    }

    private fun calculateIoU(box1: FloatArray, box2: FloatArray): Float {
        val x1 = maxOf(box1[0], box2[0])
        val y1 = maxOf(box1[1], box2[1])
        val x2 = minOf(box1[2], box2[2])
        val y2 = minOf(box1[3], box2[3])
        val intersection = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        val area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection / (area1 + area2 - intersection)
    }

    data class Result(
        val bitmap: Bitmap,
        val totalCount: Int,
        val classCounts: Map<String, Int>
    )

    fun close() {
        session.close()
        env.close()
    }
}
