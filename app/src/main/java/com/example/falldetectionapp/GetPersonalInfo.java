package com.example.falldetectionapp;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.app.ActivityManager;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.android.material.snackbar.Snackbar;
import com.google.android.material.tabs.TabLayout;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.OnProgressListener;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;

public class GetPersonalInfo extends AppCompatActivity implements SensorEventListener {
    private static final String TAG = "GetPersonalInfo";
    EditText editName, editAge, editWeight, editHeight, editPhoneNum, editMedicalCondition;
    TextView tv;
    Button btnChangePersonalAvatar, btnDone;
    ImageView imgAvatar;
    FirebaseStorage storage;
    StorageReference storageReference;
    Uri imageUri;
    private SensorManager sensorManager;
    private Sensor accelerometerSensor;
    private MediaPlayer mediaPlayer;
    private boolean isSoundPlaying = false;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_getting_emergency_contact_info);
        // Get sensor manager and accelerometer sensor
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        accelerometerSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
// Prepare the sound
        mediaPlayer = MediaPlayer.create(this, R.raw.warning_sound); // Replace with your sound file

//        if(!foregroundServiceRunning()){
//            Intent serviceIntent = new Intent(this, FallDetectionService.class);
//            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
//                startForegroundService(serviceIntent);
//            }
//        }

//        Intent serviceIntent = new Intent(this, FallDetectionService.class);
//        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
//            startForegroundService(serviceIntent);
//        }

        Database db = new Database(getApplicationContext(), "userInfo", null, 1);

        imgAvatar = findViewById(R.id.imageAvatar);

        storage = FirebaseStorage.getInstance();
        storageReference = storage.getReference();

        imgAvatar.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ChoosePicture();
            }
        });

        editName = findViewById(R.id.editTextName);
        editAge = findViewById(R.id.editTextAge);
        editWeight = findViewById(R.id.editTextWeight);
        editHeight = findViewById(R.id.editTextHeight);
        editPhoneNum = findViewById(R.id.editTextPhone);
        editMedicalCondition = findViewById(R.id.editTextMedicalConditions);

        btnChangePersonalAvatar = findViewById(R.id.buttonChangeAvatar);
        btnDone = findViewById(R.id.buttonDone);

        btnChangePersonalAvatar.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ChoosePicture();
            }
        });


        final List<String> genders = Arrays.asList("Male", "Female", "Other");
        final Spinner spinner = findViewById(R.id.spinnerGender);
        ArrayAdapter adapter = new ArrayAdapter(getApplicationContext(), android.R.layout.simple_spinner_item, genders);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);

        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                String newItem = genders.get(position);
                Toast.makeText(getApplicationContext(), "You selected: " + newItem, Toast.LENGTH_LONG).show();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {

            }
        });
        String gender = spinner.getSelectedItem().toString();

        btnDone.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(GetPersonalInfo.this, GetLocationActivity.class);

                String Name = editName.getText().toString();
                String Age = editAge.getText().toString();
                String Weight = editWeight.getText().toString();
                String Height = editHeight.getText().toString();
                String PhoneNum = editPhoneNum.getText().toString();
                String MedicalConditions = editMedicalCondition.getText().toString();
                if(Name.length() == 0 || Age.length() == 0 || Weight.length() == 0 ||
                        Height.length() == 0 || PhoneNum.length() == 0 || MedicalConditions.length() == 0){
                    Toast.makeText(getApplicationContext(), "Please fill ALL details", Toast.LENGTH_SHORT).show();
                }
                else{
                    db.doneGettingUserInfo(Name, Age, Height, Weight, PhoneNum, MedicalConditions, gender);
                    Toast.makeText(getApplicationContext(), "Success", Toast.LENGTH_SHORT).show();
                    startActivities(new Intent[]{intent});
                }
            }
        });
    }

    public boolean foregroundServiceRunning(){
        ActivityManager activityManager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
        for(ActivityManager.RunningServiceInfo service: activityManager.getRunningServices(Integer.MAX_VALUE)){
            if(FallDetectionService.class.getName().equals(service.service.getClassName())){
                return true;
            }
        }
        return false;
    }

    private void ChoosePicture() {
        Intent intent = new Intent();  // get Image from device
        intent.setType("image/");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(intent, 1);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == 1 && resultCode == RESULT_OK && data != null && data.getData() != null) {
            imageUri = data.getData();
            imgAvatar.setImageURI(imageUri);
            UploadPicture();
        }
    }

    private void UploadPicture() {

        final ProgressDialog pd = new ProgressDialog(this);
        pd.setTitle("Uploading Image...");
        pd.show();

        final String randomKey = UUID.randomUUID().toString();
        // Create a reference to "mountains.jpg"
        StorageReference mountainsRef = storageReference.child(randomKey);

        // Create a reference to 'images/mountains.jpg'
        StorageReference mountainImagesRef = storageReference.child("images/" + randomKey);

        // While the file names are the same, the references point to different files
        mountainsRef.getName().equals(mountainImagesRef.getName());    // true
        mountainsRef.getPath().equals(mountainImagesRef.getPath());    // false
        mountainImagesRef.putFile(imageUri)
        // Register observers to listen for when the download is done or if it fails
            .addOnFailureListener(new OnFailureListener() {
                @Override
                public void onFailure(@NonNull Exception exception) {
                    pd.dismiss();
                    Toast.makeText(getApplicationContext(), "Failed To Upload", Toast.LENGTH_LONG).show();
                    // Handle unsuccessful uploads
                }
            })
            .addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
                @Override
                public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {
                    pd.dismiss();
                    Snackbar.make(findViewById(android.R.id.content), "Image Uploaded", Snackbar.LENGTH_LONG).show();
                }
            })
            .addOnProgressListener(new OnProgressListener<UploadTask.TaskSnapshot>() {
                @Override
                public void onProgress(@NonNull UploadTask.TaskSnapshot snapshot) {
                    double progressPercent = (100.00 * snapshot.getBytesTransferred() / snapshot.getTotalByteCount());
                    pd.setMessage("Progress: " + (int) progressPercent + "%");
                }
            });
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            float xValue = event.values[0];
            Log.d(TAG, "X: " + xValue);
            if (xValue > 10 && !isSoundPlaying) {
                playSoundAndShowAlert();
            }
        }
    }

    private void playSoundAndShowAlert() {
        isSoundPlaying = true;
        mediaPlayer.start();

        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setMessage("X value exceeded 10. Press to stop sound.")
                .setPositiveButton("Stop", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which)
                    {
                        mediaPlayer.stop();
                        mediaPlayer.release(); // Release resources
                        mediaPlayer = MediaPlayer.create(GetPersonalInfo.this, R.raw.warning_sound); // Re-prepare for next use
                        isSoundPlaying = false;
                    }
                });
        builder.create().show();
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Not used in this example
    }
    @Override
    protected void onResume()
    {
        super.onResume();
        sensorManager.registerListener(this, accelerometerSensor, SensorManager.SENSOR_DELAY_NORMAL);
    }
    @Override
    protected void onPause()
    {
        super.onPause();
        sensorManager.unregisterListener(this);
    }
}