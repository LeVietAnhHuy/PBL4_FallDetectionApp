package com.example.falldetectionapp;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.app.ProgressDialog;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
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
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.OnProgressListener;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;

public class GetPersonalInfo extends AppCompatActivity {
    EditText editName, editAge, editWeight, editHeight, editPhoneNum, editMedicalCondition;
    TextView tv;
    Button btnChangePersonalAvatar, btnDone;
    ImageView imgAvatar;
    FirebaseStorage storage;
    StorageReference storageReference;
    Uri imageUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_getting_emergency_contact_info);

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
}