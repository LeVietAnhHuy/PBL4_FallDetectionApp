package com.example.falldetectionapp;

import android.content.ContentValues;
import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

import androidx.annotation.Nullable;

public class Database extends SQLiteOpenHelper {
    public Database(@Nullable Context context, @Nullable String name, @Nullable SQLiteDatabase.CursorFactory factory, int version) {
        super(context, name, factory, version);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        String qry1 = "create table users(name, age, weight, height, phoneNumber, medicalCondition)";
        db.execSQL(qry1);
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {

    }

    public void doneGettingUserInfo(String name, String age, String weight, String height, String phoneNumber, String medicalCondition, String gender){
        ContentValues cv = new ContentValues();
        cv.put("name", name);
        cv.put("age", age);
        cv.put("weight", weight);
        cv.put("height", height);
        cv.put("phoneNumber", phoneNumber);
        cv.put("medicalCondition", medicalCondition);
        cv.put("gender", gender);
        SQLiteDatabase db = getWritableDatabase();
        db.insert("userInfo", null, cv);
        db.close();
    }
}
