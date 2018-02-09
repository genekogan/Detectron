JSONObject json;

PImage mainImage;

void setup() {
  size(900, 600);
  
  mainImage = loadImage("/Users/gene/Downloads/15673749081_767a7fa63a_k.jpg");
  mainImage = loadImage("/Users/gene/Downloads/33887522274_eebd074106_k.jpg");
  
  json = loadJSONObject("/Users/gene/Downloads/test.json");
  println(json);
  
  int numImgs = 2;
  
  JSONArray images = json.getJSONArray("images");
  for (int i = 1; i < numImgs; i++) {
    JSONObject img = images.getJSONObject(i);
    JSONArray objects = img.getJSONArray("objects");
    for (int o = 0; o < objects.size(); o++) {
      JSONObject obj = objects.getJSONObject(o);
      JSONArray bbox = obj.getJSONArray("box");
      
      
    } 
  }
  frameRate(1);
  
  //JSONObject objects = js.getJSONObject("zebra");
  //JSONObject obj = json.getJSONArray(");
//
//  float x = bbox.getFloat("id");
//  String species = goat.getString("species");
//
//  println(id + ", " + species);
}


void drawImage(int idxImage) {
}

void draw() {
  image(mainImage, 0, 0);
  
  int numImgs = 2;
  
  JSONArray images = json.getJSONArray("images");
  for (int i = 1; i < numImgs; i++) {
    JSONObject img = images.getJSONObject(i);
    JSONArray objects = img.getJSONArray("objects");
    for (int o = 0; o < objects.size(); o++) {
      JSONObject obj = objects.getJSONObject(o);
      JSONArray bbox = obj.getJSONArray("box");
      
      String className = obj.getString("class");
      float x1 = bbox.getFloat(0);
      float y1 = bbox.getFloat(1);
      float x2 = bbox.getFloat(2);
      float y2 = bbox.getFloat(3);
      
      float score = obj.getFloat("score");
     
      noFill();
      rect(x1, y1, x2-x1, y2-y1);
      text(className, x1, y1+20);
      
      pushStyle();
      stroke(255,0,0);
      strokeWeight(2);
      
      JSONArray contours = obj.getJSONArray("contours");
      
      for (int c = 0; c < contours.size(); c++) {
        JSONArray contour = contours.getJSONArray(c);
        
        beginShape();
        for (int p = 0; p < contour.size(); p++) {
          JSONArray pt = contour.getJSONArray(p);
          float x = pt.getFloat(0);
          float y = pt.getFloat(1);
          vertex(x, y);
        }
        endShape();
      }
      
      
      popStyle();
      
    } 
  }
  
}
