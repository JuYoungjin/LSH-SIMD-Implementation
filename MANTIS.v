module MANTIS(i_Clk, i_Reset, i_Start, i_Data, i_TW, i_Key, o_fBusy, o_fDone, o_Data);


// INPUT
/*
i_Clk 	: Clock 
i_Reset : Reset current value when i_Reset is negative edge
i_Start : if i_Start is 1 then start encryption
i_Data 	: Plaintext value
i_Key	: Master key value
*/
input i_Clk, i_Reset, i_Start;
input [63:0] i_Data;
input [63:0] i_TW;
input [127:0] i_Key;

// OUTPUT
/*
o_Data	: Ciphertext value
o_fBusy	: Check State is RUN(If State is RUN, o_fBusy = 1)
o_fDone	: Check State is DONE(If State is DONE, o_fDone = 1)
*/
output wire o_fBusy, o_fDone;

output wire [63:0] o_Data;

// PARAMETER
parameter IDLE = 3'b000, RUN_HR1 = 3'b001, RUN_HR2 = 3'b010, INTER_HR = 3'b011, DONE = 3'b111;

parameter   RC1 =   64'h13198a2e03707344,
            RC2 =   64'ha4093822299f31d0,
            RC3 =   64'h082efa98ec4e6c89,
            RC4 =   64'h452821e638d01377,
            RC5 =   64'hbe5466cf34e90c6c,
            RC6 =   64'hc0ac29b7c97c50dd,
            RC7 =   64'h3f84d5b5b5470917,
            RC8 =   64'h9216d5d98979fb1b;

parameter   A   =   64'h243f6a8885a308d3;

// REGISTER
reg [63:0] 	n_Data  ,c_Data ;		// 64 bits Data register
reg [63:0]  n_TW    ,c_TW   ;       // 64 bits Tweak register
reg [127:0]	n_Key   ,c_Key  ;		// 128 bits Key register
reg [2:0] 	n_State ,c_State;	    // State register
reg [2:0] 	n_Count ,c_Count;	    // 32ROUND count register

/* WIRE */
assign o_Data	= c_Data    ^  c_TW   ; 	// o_Data = c_Data if State is DONE, otherwise o_Data = 0 
assign o_fBusy	= &c_State;
assign o_fDone	= ^c_State;

always @(posedge i_Clk or negedge i_Reset)
begin
        if(!i_Reset)
        begin
                c_State	= 0;
                c_Count	= 0;
                c_Data	= 0;
                c_TW    = 0;
                c_Key	= 0;
        end     
        else
        begin
                c_State = n_State   ;
                c_Count	= n_Count   ;
                c_Data	= n_Data    ;
                c_TW    = n_TW      ;
                c_Key	= n_Key     ;
        end     
end



// STATE REGISTER
always @(*)
begin
        n_State = c_State;
        case(c_State)
        	IDLE    :       if(i_Start)		n_State = RUN_HR1;
        	RUN_HR1 :       if(c_Count==7)	n_State = INTER_HR;	 
            INTER_HR:       if(c_Count==1)  n_State = RUN_HR2;
            RUN_HR2 :       if(c_Count==0)  n_State = DONE;
            DONE    :                     	n_State = IDLE;
        endcase 
end     

// COUNT REGISTER
always @(*)
begin
        n_Count = c_Count;
        case(c_State)   
			IDLE	:		if(i_Start)		n_Count =	0;
			RUN_HR1 :					    n_Count =	c_Count + 1;
            INTER_HR:       begin
                n_Count =   c_Count + 1;
                if(c_Count[0])  n_Count =  7;
            end
            RUN_HR2 :                       n_Count =   c_Count - 1;
            DONE	:					    n_Count =	0;
        endcase 
end     




// KEY REGISTER // Key expansion

wire    [63:0]  K0, K1, K2;
wire    [191:0] KEY;
assign  K0  =   c_Key[127:64];
assign  K1  =   c_Key[63:0];
assign  K2  =   {   K0[0], K0[63:1] }   ^   {   K1[62:0],   K1[63]  };
assign  KEY =   {   K0, K2, K1  };

always @(*)
begin
        n_Key =   c_Key;
        case(c_State)
        	IDLE	:	    if(i_Start)	    n_Key	=	i_Key;
        	RUN_HR1 :   					n_Key   =   c_Key;
			RUN_HR2 :                       n_Key   =   c_Key;
            DONE	:                       n_Key   =	0;
        endcase
end



// TWEAK REGISTER  
wire    [63:0]  HT;
wire    [63:0]  TK;

assign  HT      =  {    c_TW[39:36],    c_TW[43:40],    c_TW[ 7: 4],    c_TW[ 3: 0],
                        c_TW[63:60],    c_TW[59:56],    c_TW[55:52],    c_TW[51:48],
                        c_TW[35:32],    c_TW[15:12],    c_TW[ 9: 6],    c_TW[47:44],
                        c_TW[31:28],    c_TW[27:24],    c_TW[23:20],    c_TW[19:16] };
assign  HT_INV  =   {   c_TW[47:44],    c_TW[43:40],    c_TW[39:36],    c_TW[35:32],
                        c_TW[19:16],    c_TW[59:56],    c_TW[63:60],    c_TW[51:48],
                        c_TW[15:12],    c_TW[11: 8],    c_TW[ 7: 4],    c_TW[ 3: 0],
                        c_TW[27:24],    c_TW[23:20],    c_TW[55:52],    c_TW[51:48] };
assign  TK      =   K1  ^   HT;
assign  TK_INV  =   K1  ^   HT_INV;


always @(*)
begin
        n_TW =   c_TW;
        case(c_State)
        	IDLE	:	    if(i_Start)	    n_TW    =	i_TW;
        	RUN_HR1 :   					n_TW    =   HT;
            RUN_HR2 :                       n_TW    =   HT_INV;
            DONE	:                       n_TW    =	0;
        endcase
end



// DATA REGISTER // round function 

wire    [63:0]  SUBC;
wire    [63:0]  ADDC;
wire    [63:0]  ADDC_INV;
reg     [63:0]  RC;
wire    [63:0]  ADDTK;
wire    [63:0]  ADDTK_INV;
wire    [63:0]  PCELL;
wire    [63:0]  PCELL_INV;
wire    [63:0]  MIX;


always@(*) begin
    case(c_Count)
        0   :   RC  =   RC1;
        1   :   RC  =   RC2;
        2   :   RC  =   RC3;
        3   :   RC  =   RC4;
        4   :   RC  =   RC5;
        5   :   RC  =   RC6;
        6   :   RC  =   RC7;
        7   :   RC  =   RC8;
    endcase
end

//assign  SUBCELL_IN  =   c_Data[63:0]    ^  K0   ^   K1  ^   c_TK; 
assign  SUBCELL_IN  =   c_State == IDLE     ?   i_Data  ^   i_Key[127:64]   ^   i_Key[63:0] ^   i_TW    :   
                        c_State == RUN_HR2  ?   ADDC_INV                                                :   c_Data;


SUBCELL  SUBCELL  (SUBCELL_IN[63:60],    SUBC[:]);
SUBCELL  SUBCELL  (SUBCELL_IN[59:56],    SUBC[:]);
SUBCELL  SUBCELL  (SUBCELL_IN[55:52],    SUBC[:]);
SUBCELL  SUBCELL  (SUBCELL_IN[51:48],    SUBC[:]);
SUBCELL  SUBCELL  (SUBCELL_IN[47:44],    SUBC[:]);
SUBCELL  SUBCELL  (SUBCELL_IN[43:40],    SUBC[:]);
SUBCELL  SUBCELL  (SUBCELL_IN[39:36],    SUBC[:]);
SUBCELL  SUBCELL  (SUBCELL_IN[35:32],    SUBC[:]);
SUBCELL  SUBCELL  (SUBCELL_IN[31:28],    SUBC[:]);
SUBCELL  SUBCELL  (SUBCELL_IN[27:24],    SUBC[:]);
SUBCELL  SUBCELL  (SUBCELL_IN[23:20],    SUBC[:]);
SUBCELL  SUBCELL  (SUBCELL_IN[19:16],    SUBC[:]);
SUBCELL  SUBCELL  (SUBCELL_IN[15:12],    SUBC[:]);
SUBCELL  SUBCELL  (SUBCELL_IN[11: 8],    SUBC[:]);
SUBCELL  SUBCELL  (SUBCELL_IN[ 7: 4],    SUBC[:]);
SUBCELL  SUBCELL  (SUBCELL_IN[ 3: 0],    SUBC[:]);

/*              R           */
assign  ADDC        =   RC      ^   SUBC;
assign  ADDTK       =   ADDC    ^   TK;

assign  ADDTK_INV   =   PCELL   ^   TK_INV  ^   A;
assign  ADDC_INV    =   ADDTK_INV   ^   RC;




assign  PCELL       =   c_State == RUN_HR2  ?   {   MIX[63:60], MIX[43:40], MIX[ 3: 0], MIX[23:20],
                                                    MIX[11: 8], MIX[31:27], MIX[55:52], MIX[35:32],
                                                    MIX[19:16], MIX[ 7: 4], MIX[47:44], MIX[59:56],
                                                    MIX[39:36], MIX[51:48], MIX[27:24], MIX[15:12]  }   :

                        c_State == INTER_HR ?   SUBC    :   
                                                {   ADDTK[63:60],   ADDTK[19:16],   ADDTK[39:36],   ADDTK[11: 8],
                                                    ADDTK[23:20],   ADDTK[59:56],   ADDTK[15:12],   ADDTK[35:32],
                                                    ADDTK[43:40],   ADDTK[ 7: 4],   ADDTK[51:48],   ADDTK[31:28],
                                                    ADDTK[ 3: 0],   ADDTK[47:44],   ADDTK[27:24],   ADDTK[55:52]    };                  //64bits

//assign  PCELL_INV   =   {   ADDTK_INV[63:60],   ADDTK_INV[19:16],   ADDTK_INV[39:36],   ADDTK_INV[11: 8],
  //                          ADDTK_INV[23:20],   ADDTK_INV[59:56],   ADDTK_INV[15:12],   ADDTK_INV[35:32],
    //                        ADDTK_INV[43:40],   ADDTK_INV[ 7: 4],   ADDTK_INV[51:48],   ADDTK_INV[31:28],
      //                      ADDTK_INV[ 3: 0],   ADDTK_INV[47:44],   ADDTK_INV[27:24],   ADDTK_INV[55:52]    };

assign  MIX         =   c_State ==  RUN_HR2 ?   {   c_Data[47:44] ^ c_Data[31:28] ^ c_Data[15:12],    c_Data[43:40] ^ c_Data[27:24] ^ c_Data[11: 8],  c_Data[39:36] ^ c_Data[23:20] ^ c_Data[ 7: 4], c_Data[35:32] ^ c_Data[19:16] ^ c_Data[ 3: 0],
                                                    c_Data[63:60] ^ c_Data[31:28] ^ c_Data[15:12],    c_Data[59:56] ^ c_Data[27:24] ^ c_Data[11: 8],  c_Data[55:52] ^ c_Data[23:20] ^ c_Data[ 7: 4], c_Data[51:48] ^ c_Data[19:16] ^ c_Data[ 3: 0],
                                                    c_Data[63:60] ^ c_Data[47:44] ^ c_Data[15:12],    c_Data[59:56] ^ c_Data[43:40] ^ c_Data[11: 8],  c_Data[55:52] ^ c_Data[39:36] ^ c_Data[ 7: 4], c_Data[51:48] ^ c_Data[35:32] ^ c_Data[ 3: 0],
                                                    c_Data[63:60] ^ c_Data[47:44] ^ c_Data[31:28],    c_Data[59:56] ^ c_Data[43:40] ^ c_Data[27:24],  c_Data[55:52] ^ c_Data[39:36] ^ c_Data[23:20], c_Data[51:48] ^ c_Data[35:32] ^ c_Data[19:16]  }   :
                                                {   PCELL[47:44] ^ PCELL[31:28] ^ PCELL[15:12],    PCELL[43:40] ^ PCELL[27:24] ^ PCELL[11: 8],  PCELL[39:36] ^ PCELL[23:20] ^ PCELL[ 7: 4], PCELL[35:32] ^ PCELL[19:16] ^ PCELL[ 3: 0],
                                                    PCELL[63:60] ^ PCELL[31:28] ^ PCELL[15:12],    PCELL[59:56] ^ PCELL[27:24] ^ PCELL[11: 8],  PCELL[55:52] ^ PCELL[23:20] ^ PCELL[ 7: 4], PCELL[51:48] ^ PCELL[19:16] ^ PCELL[ 3: 0],
                                                    PCELL[63:60] ^ PCELL[47:44] ^ PCELL[15:12],    PCELL[59:56] ^ PCELL[43:40] ^ PCELL[11: 8],  PCELL[55:52] ^ PCELL[39:36] ^ PCELL[ 7: 4], PCELL[51:48] ^ PCELL[35:32] ^ PCELL[ 3: 0],
                                                    PCELL[63:60] ^ PCELL[47:44] ^ PCELL[31:28],    PCELL[59:56] ^ PCELL[43:40] ^ PCELL[27:24],  PCELL[55:52] ^ PCELL[39:36] ^ PCELL[23:20], PCELL[51:48] ^ PCELL[35:32] ^ PCELL[19:16]  };
///////////////////////////////
//
/*          R_INV           */



always@(*)
begin
        n_Data = c_Data;
        case(c_State)   
			IDLE	:		if(i_Start)		n_Data	=	i_Data;
			RUN_HR1 :                       n_Data  =   MIX;
            INTER_HR:   begin
                if(c_Count[0] == 0) n_Data  =   MIX;
                else                n_Data  =   SUBC;
            end
            RUN_HR2 :   begin 
                n_Data  =   SUBC;
                //if(c_Count == 0)    n_Data  =   SUBC    ^   K1   ^  K2   ^  A;
            end
            DONE	:					    n_Data	=	c_Data;
        endcase 
end 


endmodule

module SUBCELL(i_Data, o_Data);

input   wire    [3:0]   i_Data;
output  reg     [3:0]   o_Data;

always@(*) begin
    case(i_Data)
        4'h0    :   o_Data  =  4'hc; 
        4'h1    :   o_Data  =  4'ha;
        4'h2    :   o_Data  =  4'hd;
        4'h3    :   o_Data  =  4'h3;
        4'h4    :   o_Data  =  4'he;
        4'h5    :   o_Data  =  4'hb;
        4'h6    :   o_Data  =  4'hf;
        4'h7    :   o_Data  =  4'h7;
        4'h8    :   o_Data  =  4'h8;
        4'h9    :   o_Data  =  4'h9;
        4'ha    :   o_Data  =  4'h1;
        4'hb    :   o_Data  =  4'h5;
        4'hc    :   o_Data  =  4'h0;
        4'hd    :   o_Data  =  4'h2;
        4'he    :   o_Data  =  4'h4;
        4'hf    :   o_Data  =  4'h6;
    endcase
end

endmodule
