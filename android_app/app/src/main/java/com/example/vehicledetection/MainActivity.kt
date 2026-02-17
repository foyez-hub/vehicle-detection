package com.example.vehicledetection

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.example.vehicledetection.databinding.ActivityMainBinding
import java.io.InputStream

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var detector: YoloDetector

    private val getContent = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { processImage(it) }
    }

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap: Bitmap? ->
        bitmap?.let { runInference(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        try {
            detector = YoloDetector(this)
        } catch (e: Exception) {
            Toast.makeText(this, "Error initializing detector: ${e.message}", Toast.LENGTH_LONG).show()
        }

        binding.btnUpload.setOnClickListener {
            getContent.launch("image/*")
        }

        binding.btnCapture.setOnClickListener {
            takePicture.launch(null)
        }
    }

    private fun processImage(uri: Uri) {
        val inputStream: InputStream? = contentResolver.openInputStream(uri)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        runInference(bitmap)
    }

    private fun runInference(bitmap: Bitmap) {
        val result = detector.detect(bitmap)
        
        binding.ivResult.setImageBitmap(result.bitmap)
        binding.tvCount.text = "Total Vehicles: ${result.totalCount}"
        
        val details = result.classCounts.entries.joinToString("\n") { "${it.key}: ${it.value}" }
        binding.tvClassDetails.text = if (details.isEmpty()) "No vehicles detected" else details
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::detector.isInitialized) {
            detector.close()
        }
    }
}
