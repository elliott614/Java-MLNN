class Pair<T, V> {
	T t;
	V v;

	Pair(T t, V v) {
		this.t = t;
		this.v = v;
	}

	T first() {
		return t;
	}

	V second() {
		return v;
	}
	
	void setFirst(T t) {
		this.t = t;
	}
	
	void setSecond(V v) {
		this.v = v;
	}
	
	Boolean equals(Pair<T, V> other) {
		return (t == other.first() && v == other.second());
	}
}