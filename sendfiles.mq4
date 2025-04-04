//+------------------------------------------------------------------+
//|                                                    sendfiles.mq4 |
//|                                  Copyright 2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#import "shell32.dll"
int ShellExecuteW(int hwnd, string lpOperation, string lpFile, string lpParameters, string lpDirectory, int nShowCmd);
#import
#property copyright "Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict


extern string  SymbolName     = "XAUUSD.F";
// فایل‌های CSV
string fileM5  = SymbolName+"_M5_live.csv";
string fileM15 = SymbolName+"_M15_live.csv";
string fileM30 = SymbolName+"_M30_live.csv";
string fileH1  = SymbolName+"_H1_live.csv";

string batFilePath = "C:\\Users\\Forex\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files\\sendcsvfiles.bat";
//string batFilePath1 = "C:\\Users\\Forex\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files\\Get_answer_file.bat";
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- create timer
   EventSetTimer(1);

//---
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
   bool e5  = FileExistsCommon(fileM5);
   bool e15 = FileExistsCommon(fileM15);
   bool e30 = FileExistsCommon(fileM30);
   bool eH1 = FileExistsCommon(fileH1);

   if(e5 && e15 && e30 && eH1)
     {
      //Print("All 4 CSV files ready => Calling the BAT file to send them...");
      Sleep(1000);
      sending_file(batFilePath);

     }

  }
//+------------------------------------------------------------------+
//------------------------------------------------------------------
// تابع اجرای فایل BAT
void sending_file(string batFile)
  {
   Print("Attempting to call BAT file: ", batFile);
   int result = ShellExecuteW(0, "open", batFile, NULL, NULL, 1);

   if(result > 32)
     {
      Print("File transferred successfully using ShellExecuteW.");
     }
   else
      Print("Failed to transfer file using ShellExecuteW. Error code:", result);
  }
//+------------------------------------------------------------------+
//------------------------------------------------------------------
// تابع بررسی وجود فایل در Common\Files
bool FileExistsCommon(string fileName)
  {
   int h = FileOpen(fileName, FILE_READ|FILE_COMMON);
   if(h < 1)
      return false;
   FileClose(h);
   return true;
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
