//+------------------------------------------------------------------+
//|                                         custom_datagenerator.mq4 |
//|                                  Copyright 2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
extern int     BarCount       = 30000;
extern string  SymbolName     = "XAUUSD.F";

datetime LastBarM30 = 0;
datetime LastBarM15 = 0;
datetime LastBarM5 = 0;
datetime LastBarH1  = 0;

bool newBarM30 = false;
bool newBarH1 = false;
bool newBarM15 = false;
bool newBarM5 = false;

string fileM5  = SymbolName+"_M5_live.csv";
string fileM15 = SymbolName+"_M15_live.csv";
string fileM30 = SymbolName+"_M30_live.csv";
string fileH1  = SymbolName+"_H1_live.csv";
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- create timer
   EventSetTimer(1);
   newBarM30 = false;
   newBarH1 = false;
   newBarM15 = false;
   newBarM5 = false;
   LastBarM30 = 0;
   LastBarM15 = 0;
   LastBarM5 = 0;
   LastBarH1  = 0;
//Print(LastBarH1,LastBarM30,LastBarM15,LastBarM5);
//---
   WriteCSVFile(SymbolName, 60,    BarCount, fileH1);
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- destroy timer
   EventKillTimer();

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---


  }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
//---
   if(working_expert() == true)
     {
      Comment("Expert is off untill 01:30 AM");
      return;
     }
   newBarM30 = false;
   newBarH1 = false;
   newBarM15 = false;
   newBarM5 = false;
//Print(LastBarH1,LastBarM30,LastBarM15,LastBarM5);
   if(newBarM30 == false)
      newBarM30 = CheckNewBarM30(30, LastBarM30);
   if(newBarH1 == false)
      newBarH1 = CheckNewBarH1(60, LastBarH1);
   if(newBarM15 == false)
      newBarM15 = CheckNewBarM15(15, LastBarM15);
   if(newBarM5 == false)
      newBarM5 = CheckNewBarM5(5, LastBarM5);
   //Print("H1 = ", newBarH1,"   M30 = ",newBarM30,"   M15 = ",newBarM15,"   M5 = ",newBarM5);

   if(newBarM30 && newBarM15 && newBarM5)
     {

      //Print("M30 = ", newBarM30, "M15 = ",newBarM15, "M5 = ",newBarM5);
      datetime brokerTime = TimeCurrent();
      int minute = TimeMinute(brokerTime);
      if(minute >= 30)
        {
         Print("New candle in all time frames crated ....");
         WriteCSVFile(SymbolName,  5,    BarCount, fileM5);
         WriteCSVFile(SymbolName, 15,    BarCount, fileM15);
         WriteCSVFile(SymbolName, 30,    BarCount, fileM30);
         WriteCSVFile(SymbolName, 60,    BarCount, fileH1);
         newBarM30 = false;
         newBarH1  = false;
         newBarM15 = false;
         newBarM5  = false;
        }
      else
         if(newBarH1)
           {
            Print("New bar H1 is True");
            Print("New candle in all time frames crated ....");
            WriteCSVFile(SymbolName,  5,    BarCount, fileM5);
            WriteCSVFile(SymbolName, 15,    BarCount, fileM15);
            WriteCSVFile(SymbolName, 30,    BarCount, fileM30);
            WriteCSVFile(SymbolName, 60,    BarCount, fileH1);
            newBarM30 = false;
            newBarH1  = false;
            newBarM15 = false;
            newBarM5  = false;
           }
     }

  }
//+------------------------------------------------------------------+
bool CheckNewBarH1(int tf, datetime lastBarTime)
  {
   datetime t = iTime(SymbolName, tf, 0);
   long volume = iVolume(SymbolName, tf, 0);
   if(t <= 0 || volume == 0)
     {
      return false;
     }
   if(lastBarTime == 0)
     {
      LastBarH1 = t;
      return false;
     }
   if(t != lastBarTime)
     {
      LastBarH1 = t;
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
bool CheckNewBarM30(int tf, datetime lastBarTime)
  {
   datetime t = iTime(SymbolName, tf, 0);
   long volume = iVolume(SymbolName, tf, 0);
   if(t <= 0 || volume == 0)
     {
      return false;
     }
   if(lastBarTime == 0)
     {
      LastBarM30 = t;
      return false;
     }
   if(t != lastBarTime)
     {
      LastBarM30 = t;
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
bool CheckNewBarM15(int tf, datetime lastBarTime)
  {
   datetime t = iTime(SymbolName, tf, 0);
   long volume = iVolume(SymbolName, tf, 0);
   if(t <= 0 || volume == 0)
     {
      return false;
     }
   if(lastBarTime == 0)
     {
      LastBarM15 = t;
      return false;
     }
   if(t != lastBarTime)
     {
      LastBarM15 = t;
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
bool CheckNewBarM5(int tf, datetime lastBarTime)
  {
   datetime t = iTime(SymbolName, tf, 0);
   long volume = iVolume(SymbolName, tf, 0);
   if(t <= 0 || volume == 0)
     {
      return false;
     }
   if(lastBarTime == 0)
     {
      LastBarM5 = t;
      return false;
     }
   if(t != lastBarTime)
     {
      LastBarM5 = t;
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+



//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool WriteCSVFile(string symbol, int tf, int barsToWrite, string filename)
  {
//Print("TF = ", tf);
   int handle = FileOpen(filename, FILE_WRITE|FILE_COMMON);
   if(handle < 1)
     {
      Print("Cannot open file for writing: ", filename);
      return false;
     }
   FileWriteString(handle, "time,open,high,low,close,volume\n");
   datetime t;
   for(int i = barsToWrite; i >= 1; i--)
     {
      t = iTime(symbol, tf, i)+ tf * 60;

      double   o = iOpen(symbol, tf, i);
      double   h = iHigh(symbol, tf, i);
      double   l = iLow(symbol, tf, i);
      double   c = iClose(symbol, tf, i);
      long     v = iVolume(symbol, tf, i);

      string line = FormatTimeYMDHM(t)+","+
                    DoubleToString(o,6)+","+
                    DoubleToString(h,6)+","+
                    DoubleToString(l,6)+","+
                    DoubleToString(c,6)+","+
                    (string)v+"\n";
      FileWriteString(handle, line);
     }

   FileClose(handle);
   return true;

  }
//+------------------------------------------------------------------+
string FormatTimeYMDHM(datetime when)
  {
   int yy = TimeYear(when),
       mm = TimeMonth(when),
       dd = TimeDay(when),
       HH = TimeHour(when),
       MI = TimeMinute(when);
   return StringFormat("%04d.%02d.%02d %02d:%02d", yy, mm, dd, HH, MI);
  }
//+------------------------------------------------------------------+
bool working_expert()
  {
   datetime brokerTime = TimeCurrent(); // زمان جاری بروکر
   int hour = TimeHour(brokerTime);     // استخراج ساعت از زمان بروکر
   int minute = TimeMinute(brokerTime); // استخراج دقیقه از زمان بروکر

   if((hour == 22 && minute >= 30) ||  // از 22:30 تا 23:59
      (hour == 23) ||                 // کل ساعت 23
      (hour == 0) ||                  // کل ساعت 00:00 تا 00:59
      (hour == 1 && minute <= 30))     // از 01:00 تا 01:30
     {
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
