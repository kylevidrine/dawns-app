export interface Scan {
  id: string;
  image: string;
  timestamp: number;
  sent: boolean;
}

export interface Point {
  x: number;
  y: number;
}
