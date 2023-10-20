package com.example.falldetectionapp;

import android.app.AlertDialog;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.MediaPlayer;
import android.os.Build;
import android.os.IBinder;
import android.util.Log;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.Nullable;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class FallDetectionService extends Service implements SensorEventListener {
    private static final String TAG = "FallDetectionService";
    private SensorManager sensorManager;
    Sensor accelerometer, gyroscope, orientation;
    Module module;
    Queue<ArrayList<Float>> incoming_row_data = new LinkedList<>();
    ArrayList<Float> row_data = new ArrayList<>();
    Float[] input_data = new Float[800* 9];
    float[] input_data_float = new float[800* 9];
    ArrayList<Float> temp = new ArrayList<>();
    List<String> labels;
//    float[] scores = output_tensor.getDataAsFloatArray();
    float maxScore = -Float.MAX_VALUE;
    int maxScoreIdx = 0;
    Tensor input_tensor = null;
    Tensor output_tensor = null;
    float[] scores = null;
    AlertDialog.Builder builder;
    double threshold = 0;
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.d(TAG, "OnStartCommand: Initializing Sensor Services");

        labels = LoadLabels("labels.txt");
        LoadTorchModule("best_mobile.pt");
//        Log.d(TAG, "OnStartCommand: Initializing Model");
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        orientation = sensorManager.getDefaultSensor(Sensor.TYPE_ORIENTATION);

        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_FASTEST);
//        Log.d(TAG, "OnStartCommand: Initializing Accelerometer");
        sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_FASTEST);
        sensorManager.registerListener(this, orientation, SensorManager.SENSOR_DELAY_FASTEST);

        String classResult = labels.get(maxScoreIdx);
//            Log.d(TAG, "Result: " + classResult);
//            Log.d(TAG, "OnStartCommand: Registered accelerometer listener");
        final String CHANNEL_ID = "Foreground Service ID";
        NotificationChannel channel = null;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            channel = new NotificationChannel(
                    CHANNEL_ID,
                    CHANNEL_ID,
                    NotificationManager.IMPORTANCE_LOW
            );
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {

            getSystemService(NotificationManager.class).createNotificationChannel(channel);

        }
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            Notification.Builder notification = new Notification.Builder(this, CHANNEL_ID)
                    .setContentText("Fall Detection Service is running")
                    .setContentTitle("Service enabled")
                    .setSmallIcon(R.drawable.ic_launcher_background);
            startForeground(1001, notification.build());
        }
//                        .setSmallIcon()
        return super.onStartCommand(intent, flags, startId);
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent){
        return null;
    }

    @Override
    public void onAccuracyChanged (Sensor sensor,int accuracy){

    }

    @Override
    public void onSensorChanged (SensorEvent sensorEvent){
//        Log.d(TAG, "OnStartCommand: Initializing acc vakye");
        threshold = 0;
        Sensor sensor = sensorEvent.sensor;
        boolean detected = false;
        if (sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
//            Log.d(TAG, "OnStartCommand: Initializing acc vakye");
//            Log.d(TAG, "Accelerator: X: " + sensorEvent.values[0] + "| Y: " + sensorEvent.values[1] + "| Z: " + sensorEvent.values[2]);
            row_data.add(sensorEvent.values[0]);
            row_data.add(sensorEvent.values[1]);
            row_data.add(sensorEvent.values[2]);
            threshold = Math.sqrt(Math.pow(sensorEvent.values[0], 2) + Math.pow(sensorEvent.values[1], 2) + Math.pow(sensorEvent.values[2], 2));
//            Log.d(TAG, "threshold: " + threshold);
            if(threshold > 3*9.8)
            {
                SensorManager sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
                sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_NORMAL, 1000000);
                Toast.makeText(getApplicationContext(), "Warning!!!", Toast.LENGTH_SHORT).show();
                MediaPlayer mediaPlayer = MediaPlayer.create(this, R.raw.warning_sound);
                mediaPlayer.start();
//                builder = new AlertDialog.Builder(this);
//                builder.setCancelable(true);
//                builder.setMessage("Warning!!!");
//                builder.setTitle("You need help!!!");
//
//                builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
//                    @Override
//                    public void onClick(DialogInterface dialog, int which) {
//                        dialog.cancel();
//                    }
//                });
//
//                builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
//                    @Override
//                    public void onClick(DialogInterface dialog, int which) {
//                        dialog.cancel();
//                    }
//                });
//                builder.show();
                threshold = 0;
            }

//            Log.d(TAG, "OnStartCommand: Initializing acc vakye");
        } else if (sensor.getType() == Sensor.TYPE_GYROSCOPE) {
//            Log.d(TAG, "Gyroscope: X: " + sensorEvent.values[0] + "| Y: " + sensorEvent.values[1] + "| Z: " + sensorEvent.values[2]);
            row_data.add(sensorEvent.values[0]);
            row_data.add(sensorEvent.values[1]);
            row_data.add(sensorEvent.values[2]);
        } else {
//            Log.d(TAG, "Orientation: X: " + sensorEvent.values[0] + "| Y: " + sensorEvent.values[1] + "| Z: " + sensorEvent.values[2]);
            row_data.add(sensorEvent.values[0]);
            row_data.add(sensorEvent.values[1]);
            row_data.add(sensorEvent.values[2]);
        }
//        Log.d(TAG, "row_data: " + row_data);
//        Log.d(TAG, "row_data_size: " + row_data.size());

//        if(row_data.size() == 800*9){
//            if(detected){
//                incoming_row_data.add(row_data);
//                Log.d(TAG, "incoming_row_data: " + incoming_row_data);
////                Log.d(TAG, "row_data: " + row_data);
////                Log.d(TAG, "row_data: " + incoming_data);
////            Log.d(TAG, "size: " + row_data.size());
//
//                for(int i = 0; i < 800*9; i++)
//                {
////                Log.d(TAG, "for(int i = 0; i < 800*9; i++): start!" + input_data[i]);
//                    input_data_float[i] = row_data.get(i).floatValue();
////                Log.d(TAG, "for(int i = 0; i < 800*9; i++): Done!");
//                }
//                Log.d(TAG, "InputTensor: Start ");
//                input_tensor = Tensor.fromBlob(input_data_float, new long[]{1, 9, 800});
//                Log.d(TAG, "InputTensor: DONE! ");
//                output_tensor = module.forward(IValue.from(input_tensor)).toTensor();
//                Log.d(TAG, "OnStartCommand: Registered accelerometer listener---------------------------------------");
//                scores = output_tensor.getDataAsFloatArray();
//                Log.d(TAG, "Scores: " + scores);
//
////            float maxScore = -Float.MAX_VALUE;
//                int maxScoreIdx = 0;
//                for (int i = 0; i < 16; i++) {
//                    if (scores[i] > maxScore) {
//                        maxScore = scores[i];
//                        maxScoreIdx = i;
//                        Log.d(TAG, "ID_labels: " + maxScore);
//                    }
//                }
//                Log.d(TAG, "Result: " + labels.get(maxScoreIdx));
//
//                row_data.clear();
//                scores = null;
//            }
//        }

//        if(incoming_row_data.size() > 0)
//        {
////            Log.d(TAG, "Geting to if incoming: Start " + incoming_row_data);
//            ArrayList<Float>  test = new ArrayList<>();
//            test = incoming_row_data.poll();
//
//            input_data = temp.toArray(input_data);
//
//
//            for(int i = 0; i < 800*9; i++)
//            {
////                Log.d(TAG, "for(int i = 0; i < 800*9; i++): start!" + input_data[i]);
//                input_data_float[i] = input_data[i].floatValue();
////                Log.d(TAG, "for(int i = 0; i < 800*9; i++): Done!");
//            }
//            Log.d(TAG, "InputTensor: Start ");
//            input_tensor = Tensor.fromBlob(input_data_float, new long[]{800, 9, 1});
//            Log.d(TAG, "InputTensor: DONE! ");
//            output_tensor = module.forward(IValue.from(input_tensor)).toTensor();
//            Log.d(TAG, "OnStartCommand: Registered accelerometer listener---------------------------------------");
//            scores = output_tensor.getDataAsFloatArray();
////            float maxScore = -Float.MAX_VALUE;
////            int maxScoreIdx = -1;
//            for (int i = 0; i < 19; i++) {
//                if (scores[i] > maxScore) {
//                    maxScore = scores[i];
//                    maxScoreIdx = i;
//                }
//            }
//            Log.d(TAG, "Result: " + labels.get(maxScoreIdx));
    }
//        if (incoming_data.size() > 800) {
//            for (int i = 0; i < 800; i++) {
//                List<Float> incoming_data_row = incoming_data.removeFirst();
//                for (int j = 0; j < 9; j++) {
//                    input_data[index] = incoming_data_row.get(j);
//                    index++;
//                }
//            }

//                Log.d(TAG, "Result: " + labels.get(maxScoreIdx));


    void LoadTorchModule (String fileName){
        File modelFile = new File(this.getFilesDir(), fileName);

        try {
            if (!modelFile.exists()) {
                InputStream inputStream = getAssets().open(fileName);
                FileOutputStream outputStream = new FileOutputStream(modelFile);
                byte[] buffer = new byte[2048];
                int byteRead = -1;
                while ((byteRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, byteRead);
                }
                inputStream.close();
                outputStream.close();
            }
            module = LiteModuleLoader.load(modelFile.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    List<String> LoadLabels(String fileName){
        List<String> labels = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(fileName)));
            String line;
            while((line = br.readLine()) != null){
                labels.add(line);
            }
        } catch(IOException e){
            e.printStackTrace();
        }

        return labels;
    }

}


