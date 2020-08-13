public void NULL_DEREFERENCE(){
			int[] row = { cells[i * 4], cells[i * 4 + 1], cells[i * 4 + 2], cells[i * 4 + 3] };

			outputCells[i * 4] = MULT2[row[0]] ^ MULT3[row[1]] ^ row[2] ^ row[3];
			outputCells[i * 4 + 1] = row[0] ^ MULT2[row[1]] ^ MULT3[row[2]] ^ row[3];
			outputCells[i * 4 + 2] = row[0] ^ row[1] ^ MULT2[row[2]] ^ MULT3[row[3]];

}