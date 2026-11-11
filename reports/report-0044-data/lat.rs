use std::time::Instant;
fn main(){
  let mut s:u64=88172645463325252;
  let mut rnd=||{s^=s<<13;s^=s>>7;s^=s<<17;s};
  for mb in [1usize,4,8,32,128,512,2048]{
    let lines=mb*(1<<20)/64;
    let mut perm:Vec<u32>=(0..lines as u32).collect();
    for i in (1..lines).rev(){let j=(rnd()%(i as u64+1)) as usize;perm.swap(i,j);}
    let mut nxt=vec![[0u32;16];lines];
    for i in 0..lines{nxt[perm[i] as usize][0]=perm[(i+1)%lines];}
    let mut p=perm[0] as usize; let m=3_000_000;
    let t=Instant::now();
    for _ in 0..m{p=nxt[p][0] as usize;}
    println!("{mb} MiB: {:.1} ns/hop (p={p})",t.elapsed().as_secs_f64()*1e9/m as f64);
  }
}
