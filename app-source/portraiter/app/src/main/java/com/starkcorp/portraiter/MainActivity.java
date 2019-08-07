package com.starkcorp.portraiter;

import android.app.Dialog;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.media.Image;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Handler;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Toast;


import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLEncoder;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    public static final int SELECT_PICTURE = 100;
    private EditText name;
    private EditText password;
    private EditText email;
    private Spinner spinner;
    private ImageView profilepic;
    private Button btn,btn1;

    private static final String TAG = "Hello";
    String SERVER = "http://139.59.67.61/server.php";
    ProgressDialog progressDialog;
    ByteArrayOutputStream byteArrayOutputStream;
    Dialog dialog;
    boolean check = true;
    public static String nametext;
    private static String emailtext;
    private static String Countrytext;
    private static String passwordtext;
    private Bitmap imagenew;
    private String filename;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        profilepic=(ImageView)findViewById(R.id.imageView);
        btn=(Button)findViewById(R.id.button);
        btn1=(Button)findViewById(R.id.button1);
    }

    public void onClick(View v){
         ChooseImage();

    }

    public void onClick1(View v){
        Bitmap image=((BitmapDrawable)profilepic.getDrawable()).getBitmap();

        new Upload(image,filename).execute();
    }


    void ChooseImage() {
        Intent i = new Intent();
        i.setType("image/*");
        i.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(i, "Select Picture"), SELECT_PICTURE);
    }


    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (resultCode == RESULT_OK) {
            if (requestCode == SELECT_PICTURE) {
                Uri uri = data.getData();
                String path=uri.getPath();
                if (uri != null) {
                    profilepic.setImageURI(uri);
                    File file=new File(path);
                    filename=file.getName();
                    filename=filename.substring(filename.lastIndexOf(':')+1);
                    filename="file"+filename;





                }
            }
        }
    }


    private String HashMaptoURL(HashMap<String, String> data) throws UnsupportedEncodingException {
        StringBuilder result = new StringBuilder();
        boolean first = true;
        for (Map.Entry<String, String> entry : data.entrySet()) {
            if (first)
                first = false;
            else
                result.append("&");
            result.append(URLEncoder.encode(entry.getKey(), "UTF-8"));
            result.append("=");
            result.append(URLEncoder.encode(entry.getValue(), "UTF-8"));

        }
        return result.toString();
    }

    private class Upload extends AsyncTask<Void, Void, String> {
        private Bitmap image;
        private String imagename;

        public Upload(Bitmap image, String name) {
            this.image = image;
            this.imagename = name;
        }

        @Override
        protected void onPostExecute(String o) {
            super.onPostExecute(o);
            progressDialog.dismiss();
            progressDialog=ProgressDialog.show(MainActivity.this,"Processing","Wait",false,false);
            final Handler handler = new Handler();
            handler.postDelayed(new Runnable() {
                @Override
                public void run() {
            class GETJSON extends AsyncTask<String,Void,String>{
                @Override
                protected void onPostExecute(String s) {
                    super.onPostExecute(s);
                    profilepic.setImageBitmap(imagenew);
                    ContextWrapper wrapper = new ContextWrapper(getApplicationContext());
                    File file = wrapper.getDir("Images",MODE_PRIVATE);
                    String pat=file.getPath();
                    file = new File(file, filename+".jpg");
                    try{
                        OutputStream stream=null;
                        stream=new FileOutputStream(file);
                        imagenew.compress(Bitmap.CompressFormat.JPEG,100,stream);
                        stream.flush();
                        stream.close();
                        Toast.makeText(MainActivity.this,pat,Toast.LENGTH_LONG).show();
                    }catch(IOException e){
                        e.printStackTrace();
                    }
                    progressDialog.dismiss();


                }

                @Override
                protected void onPreExecute() {
                    super.onPreExecute();
                    progressDialog.dismiss();
                    progressDialog=ProgressDialog.show(MainActivity.this,"Downloading","Wait",false,false);

                }

                @Override
                protected String doInBackground(String... params) {


                                try {
                                    //Do something after 20 seconds
                                    String imgpath = "http://139.59.67.61/"+filename+"_seg.png";
                                    URL imgUrl = new URL(imgpath);
                                    imagenew = BitmapFactory.decodeStream(imgUrl.openConnection().getInputStream());
                                }catch(IOException e){
                                    Log.i("Tag",e.getMessage());
                                }


                return null;
                }
            }
            new GETJSON().execute();
                }
            }, 60000);
        }

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            progressDialog = ProgressDialog.show(MainActivity.this, "Image is Uploading", "Please Wait", false, false);

        }

        @Override
        protected String doInBackground(Void... params) {
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();

            image.compress(Bitmap.CompressFormat.JPEG, 50, byteArrayOutputStream);
            String EncodeImage = Base64.encodeToString(byteArrayOutputStream.toByteArray(), Base64.DEFAULT);
            HashMap<String, String> details = new HashMap<>();
            details.put("name", imagename);
            details.put("image", EncodeImage);

            try {
                String DatatoSend = HashMaptoURL(details);
                String Response = Request.post(SERVER, DatatoSend);
                return Response;
            } catch (Exception e) {
                e.printStackTrace();
                Log.e(TAG, "ERROR  " + e);
                return null;
            }
        }
    }




}